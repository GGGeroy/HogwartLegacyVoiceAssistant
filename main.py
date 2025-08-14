import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import pyautogui
from collections import deque
from datetime import datetime

# ==============================================================================
# ФИНАЛЬНАЯ КОНФИГУРАЦИЯ
# ==============================================================================
MODEL_PATH = 'best_spell_model.pth'
CONFIG_PATH = 'spell_config.json'
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.2
LOOP_INTERVAL = 0.08  # Минимальный интервал для максимальной отзывчивости

# --- НАСТРОЙКА ОТЗЫВЧИВОСТИ ---
# Установите True для максимальной скорости реакции (срабатывание от 1 уверенного слова).
# Установите False для максимальной надежности (требуется 2+ попадания).
RESPONSIVE_MODE = True

# --- Параметры для режимов (можно тонко настроить) ---
# Для стандартного режима (RESPONSIVE_MODE = False) (Раскоммнтируй код ниже 3 строчки)
#CONFIDENCE_STABLE = 0.75
#HITS_STABLE = 2
#BUFFER_SIZE_STABLE = 5

# Для отзывчивого режима (RESPONSIVE_MODE = True)
CONFIDENCE_RESPONSIVE = 0.85  # Немного выше, т.к. срабатываем от 1 слова
HITS_RESPONSIVE = 1
BUFFER_SIZE_RESPONSIVE = 3  # Буфер меньше, т.к. не накапливаем хиты

# ==============================================================================
# КОМПОНЕНТЫ ИЗ СКРИПТА ОБУЧЕНИЯ (без изменений)
# ==============================================================================
spell_pronunciation_map = {'Акцио': 'а+кцио', 'авада_кедавра': 'ав+ада кед+авра',
                           'аресто_моментум': 'ар+есто мом+энтум', 'бомбарда': 'бомб+арда',
                           'вингардиум_левиоса': 'вингардиум леви+оса', 'глациус': 'гл+ациус',
                           'делюминация': 'делюмин+ация', 'депульсо': 'дэп+ульсо', 'десцендо': 'дэсц+ендо',
                           'диффиндо': 'дифф+индо', 'империус': 'имп+ериус', 'инсендио': 'инс+ендио',
                           'конфринго': 'конфр+инго', 'круцио': 'кр+уцио', 'левиосо': 'леви+осо', 'люмос': 'л+юмос',
                           'протего': 'прот+эго', 'ревелио': 'рев+елио', 'репаро': 'реп+аро',
                           'трансформация': 'трансформ+ация', 'флиппендо': 'флипп+ендо',
                           'экспеллиармус': 'экспелли+армус'}


def extract_advanced_features(audio, sr=16000, n_mels=128, n_frames=75):
    try:
        hop_length = int(len(audio) / n_frames) if n_frames > 0 else 512
        if hop_length == 0: hop_length = 1
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048, fmin=20,
                                             fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if log_mel.shape[1] > n_frames:
            log_mel = log_mel[:, :n_frames]
        else:
            log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])), 'constant')
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        delta = librosa.feature.delta(log_mel)
        delta2 = librosa.feature.delta(log_mel, order=2)
        return torch.from_numpy(np.stack([log_mel, delta, delta2])).float()
    except Exception:
        return None


class AdvancedSpellCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 128, 75)):
        super(AdvancedSpellCNN, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
                                         nn.MaxPool2d(2), nn.Dropout(0.2), nn.Conv2d(64, 128, 3, padding=1),
                                         nn.BatchNorm2d(128), nn.ELU(), nn.MaxPool2d(2),
                                         nn.Conv2d(128, 256, 3, padding=2, dilation=2), nn.BatchNorm2d(256), nn.SiLU(),
                                         nn.AdaptiveMaxPool2d((4, 4)))
        with torch.no_grad(): self.flatten_size = self.conv_blocks(torch.zeros(1, *input_shape)).view(1, -1).shape[1]
        self.head = nn.Sequential(nn.Linear(self.flatten_size, 512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(0.4),
                                  nn.Linear(512, num_classes))

    def forward(self, x): return self.head(self.conv_blocks(x).view(x.size(0), -1))


# ==============================================================================
# ОСНОВНАЯ ЛОГИКА
# ==============================================================================
def load_resources():
    print("--- Загрузка ресурсов ---")
    if not os.path.exists(MODEL_PATH): print(f"ОШИБКА: Файл модели не найден: {MODEL_PATH}"); exit()
    if not os.path.exists(CONFIG_PATH): print(f"ОШИБКА: Файл конфигурации не найден: {CONFIG_PATH}"); exit()
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    class_names = ['unknown'] + sorted(list(spell_pronunciation_map.keys()))
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedSpellCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Модель '{MODEL_PATH}' и конфиг '{CONFIG_PATH}' загружены на {device}.")
    return model, config, idx_to_class, device


def listen_and_recognize(model, config, idx_to_class, device):
    # Выбираем параметры в зависимости от режима
    if RESPONSIVE_MODE:
        conf_thresh, required_hits, buffer_size = CONFIDENCE_RESPONSIVE, HITS_RESPONSIVE, BUFFER_SIZE_RESPONSIVE
        print("--- РЕЖИМ: ОТЗЫВЧИВЫЙ (срабатывание от 1 слова) ---")
    else:
        conf_thresh, required_hits, buffer_size = CONFIDENCE_STABLE, HITS_STABLE, BUFFER_SIZE_STABLE
        print("--- РЕЖИM: СТАБИЛЬНЫЙ (срабатывание от 2+ слов) ---")

    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    audio_buffer = np.zeros(int((CHUNK_DURATION + 0.5) * SAMPLE_RATE), dtype=np.float32)
    prediction_buffer = deque(maxlen=buffer_size)

    # Словарь для отслеживания времени последнего срабатывания КАЖДОГО заклинания
    last_spell_times = {spell: 0 for spell in idx_to_class.values()}

    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer
        shift = len(indata)
        audio_buffer = np.roll(audio_buffer, -shift)
        audio_buffer[-shift:] = indata.flatten()

    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, dtype='float32')
    with stream:
        print("\nПрослушивание началось... Говорите заклинания. (Ctrl+C для выхода)")
        try:
            while True:
                chunk = audio_buffer[-chunk_samples:]
                features = extract_advanced_features(chunk)
                if features is None: time.sleep(LOOP_INTERVAL); continue

                with torch.no_grad():
                    outputs = model(features.unsqueeze(0).to(device))
                    confidence, pred_idx = torch.max(torch.softmax(outputs, dim=1), 1)

                pred_class, confidence = idx_to_class[pred_idx.item()], confidence.item()
                prediction_buffer.append(pred_class)

                # --- УСОВЕРШЕНСТВОВАННАЯ ЛОГИКА СРАБАТЫВАНИЯ ---
                spell_cooldown = config['spells'].get(pred_class, {}).get('cooldown', 0)

                if (pred_class != 'unknown' and
                        confidence >= conf_thresh and
                        prediction_buffer.count(pred_class) >= required_hits and
                        time.time() - last_spell_times[pred_class] > spell_cooldown):

                    hotkey = config['spells'].get(pred_class, {}).get('hotkey')
                    if hotkey:
                        pyautogui.press(hotkey)

                        # ОБНОВЛЕНИЕ ВРЕМЕНИ И ВЫВОД В КОНСОЛЬ
                        last_spell_times[pred_class] = time.time()
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(
                            f"[{timestamp}] Активировано: {pred_class.upper():<20} -> Нажата клавиша: '{hotkey}' (Кулдаун: {spell_cooldown}с)")

                        prediction_buffer.clear()  # Очищаем буфер для предотвращения двойных срабатываний

                time.sleep(LOOP_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n--- Распознавание остановлено пользователем. ---")


if __name__ == "__main__":
    model, config, idx_to_class, device = load_resources()
    listen_and_recognize(model, config, idx_to_class, device)

