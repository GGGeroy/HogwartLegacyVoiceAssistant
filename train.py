import os
import random
import numpy as np
import pandas as pd # Не используется, можно удалить
import librosa
import noisereduce as nr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchaudio
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import shutil # Добавлен для надежного удаления временных файлов

# --- ПУТИ К ДАННЫМ ---
# Путь к сгенерированному аудио Silero TTS
spell_data_root = r"C:\Users\Admin\PycharmProjects\SPELL_NEIRO\audio\raw_tts_data(Silero)"

# Путь к тестовым данным (не используется напрямую в этом скрипте, но оставлен для информации)
test_data_root = r"C:\Users\Admin\PycharmProjects\SPELL_NEIRO\audio\test"

# Путь к папке с русской речью для класса 'unknown'
unknown_data_root = r"C:\Users\Admin\PycharmProjects\SPELL_NEIRO\audio\cv-corpus-22.0-2025-06-20-ru\cv-corpus-22.0-2025-06-20\ru\clips"

# --- ОБЩИЕ ПАРАМЕТРЫ ---
MAX_UNKNOWN_SAMPLES = 100000 # Ограничение на количество сэмплов 'unknown' для полного обучения

# --- ПАРАМЕТРЫ ДЛЯ ЛАЙТ-ВЕРСИИ ---
LIGHT_MODE = False # Установите False для полного обучения
MAX_SAMPLES_PER_SPELL_CLASS_LIGHT = 100 # Количество сэмплов на каждое заклинание в лайт-режиме
MAX_UNKNOWN_SAMPLES_LIGHT = 500 # Количество сэмплов 'unknown' в лайт-режиме
NUM_EPOCHS_LIGHT = 3 # Количество эпох в лайт-режиме

# Аугментация
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
])

# ==============================================================================
# СЛОВАРЬ ДЛЯ ЗАКЛИНАНИЙ (должен быть такой же, как в скрипте генерации TTS)
# Этот словарь используется для определения того, какие папки считаются заклинаниями
# и для создания согласованной карты меток.
# ==============================================================================
spell_pronunciation_map = {
    'Акцио': 'а+кцио',
    'авада_кедавра': 'ав+ада кед+авра',
    'аресто_моментум': 'ар+есто мом+энтум',
    'бомбарда': 'бомб+арда',
    'вингардиум_левиоса': 'вингардиум леви+оса',
    'глациус': 'гл+ациус',
    'делюминация': 'делюмин+ация',
    'депульсо': 'дэп+ульсо',
    'десцендо': 'дэсц+ендо',
    'диффиндо': 'дифф+индо',
    'империус': 'имп+ериус',
    'инсендио': 'инс+ендио',
    'конфринго': 'конфр+инго',
    'круцио': 'кр+уцио',
    'левиосо': 'леви+осо',
    'люмос': 'л+юмос',
    'протего': 'прот+эго',
    'ревелио': 'рев+елио',
    'репаро': 'реп+аро',
    'трансформация': 'трансформ+ация',
    'флиппендо': 'флипп+ендо',
    'экспеллиармус': 'экспелли+армус'
}


def advanced_augment(audio, sr):
    if random.random() > 0.3:
        audio = augmenter(samples=audio, sample_rate=sr)
    if random.random() > 0.5:
        try:
            # Изменен параметр 'prop_decrease' для более мягкого шумоподавления
            audio = nr.reduce_noise(y=audio, sr=sr, verbose=False, prop_decrease=0.9)
        except Exception:
            pass  # Игнорируем ошибки, если шумоподавление не удалось
    if random.random() > 0.2:
        # Добавление белого шума с случайным SNR
        noise = torch.randn(1, len(audio))
        audio = torchaudio.functional.add_noise(
            torch.from_numpy(audio).unsqueeze(0),
            noise,
            snr=torch.tensor([random.uniform(5, 20)])
        ).numpy()[0]
    return audio


def load_audio_advanced(file_path, target_sr=16000, duration=1.2, augment=False):
    try:
        audio, sr = torchaudio.load(file_path)
        if audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True)  # Преобразование стерео в моно
        audio = audio.numpy()[0]
        if sr != target_sr: audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        audio = librosa.util.normalize(audio) * 0.9  # Нормализация аудио
        target_samples = int(target_sr * duration)
        if len(audio) > target_samples:
            # Если аудио длиннее, берем случайный фрагмент
            start = random.randint(0, len(audio) - target_samples)
            audio = audio[start:start + target_samples]
        else:
            # Если аудио короче, дополняем нулями
            audio = np.pad(audio, (0, max(0, target_samples - len(audio))), 'constant')
        if augment:
            audio = advanced_augment(audio, target_sr)
        return audio
    except Exception as e:
        # print(f"WARNING: Could not load or preprocess audio {file_path}: {e}") # Отладочное сообщение
        return None


def extract_advanced_features(audio, sr=16000, n_mels=128, n_frames=75):
    try:
        # Улучшенное вычисление hop_length для обеспечения n_frames
        hop_length = int(
            len(audio) / n_frames) if n_frames > 0 else 512  # Избегаем деления на ноль, используем разумное значение по умолчанию
        if hop_length == 0: hop_length = 1  # Минимальное значение hop_length

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048, fmin=20,
                                             fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if log_mel.shape[1] > n_frames:
            log_mel = log_mel[:, :n_frames]
        else:
            log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])), 'constant')
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)  # Нормализация по Z-score
        delta = librosa.feature.delta(log_mel)
        delta2 = librosa.feature.delta(log_mel, order=2)
        features = np.stack([log_mel, delta, delta2])
        return torch.from_numpy(features).float()
    except Exception as e:
        # print(f"WARNING: Could not extract features: {e}") # Отладочное сообщение
        return None


class AdvancedSpellCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 128, 75)):
        super(AdvancedSpellCNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=2, dilation=2), nn.BatchNorm2d(256), nn.SiLU(), nn.AdaptiveMaxPool2d((4, 4))
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            self.flatten_size = self.conv_blocks(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(self.flatten_size, 512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class DiskDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, temp_dir='temp_features'):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        self.feature_paths = []
        self.labels = []

        # Используем tqdm для прогресса сохранения на диск
        print(f"Сохранение фич на диск ({len(features)} сэмплов)...")
        for idx, (feature, label) in enumerate(tqdm(zip(features, labels), total=len(features))):
            feature_path = os.path.join(temp_dir, f'feature_{idx}.pt')
            torch.save(feature, feature_path)
            self.feature_paths.append(feature_path)
            self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Убедимся, что labels - это tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        return feature, self.labels[idx]

    def cleanup(self):
        # Отдельная функция для надежной очистки временных файлов
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Временная директория '{self.temp_dir}' успешно удалена.")
            except Exception as e:
                print(f"Ошибка при удалении временной директории '{self.temp_dir}': {e}")


def load_data():
    print("--- Начало загрузки данных ---")
    all_features = []
    all_labels = []

    # Создаем фиксированную карту меток для согласованности
    # 'unknown' всегда будет 0
    # Остальные заклинания сортируются по алфавиту для стабильного индексирования
    class_names = ['unknown'] + sorted(list(spell_pronunciation_map.keys()))
    label_map = {name: i for i, name in enumerate(class_names)}
    idx_to_class_name = {i: name for i, name in enumerate(class_names)}

    print("Определенная карта меток (Label Map):",
          {k: v for k, v in sorted(label_map.items(), key=lambda item: item[1])})

    # --- Загрузка заклинаний из Silero ---
    print(f"\nЗагрузка заклинаний из '{spell_data_root}'...")
    # Получаем список всех поддиректорий внутри spell_data_root
    all_subdirs = [d for d in os.listdir(spell_data_root) if os.path.isdir(os.path.join(spell_data_root, d))]

    # Отфильтровываем только те, которые соответствуют нашим заклинаниям
    recognized_spell_subdirs = [s for s in all_subdirs if s in spell_pronunciation_map]

    if not recognized_spell_subdirs:
        print(
            f"!!! ПРЕДУПРЕЖДЕНИЕ: В '{spell_data_root}' не найдено поддиректорий, соответствующих известным заклинаниям из `spell_pronunciation_map`.")
        print(f"Проверьте, что TTS генератор создал папки с заклинаниями внутри '{spell_data_root}'.")

    for spell_name in recognized_spell_subdirs:
        spell_label = label_map[spell_name]
        spell_dir_path = os.path.join(spell_data_root, spell_name)

        # Перебираем файлы внутри каждой папки заклинаний
        files_in_spell_dir = [f for f in os.listdir(spell_dir_path) if f.lower().endswith(('.wav', '.mp3'))]
        random.shuffle(files_in_spell_dir) # Перемешиваем, чтобы взять случайную подвыборку

        # Применяем ограничение, если включен LIGHT_MODE
        if LIGHT_MODE:
            files_to_load_for_spell = files_in_spell_dir[:MAX_SAMPLES_PER_SPELL_CLASS_LIGHT]
        else:
            files_to_load_for_spell = files_in_spell_dir

        for file in tqdm(files_to_load_for_spell, desc=f"Класс '{spell_name}'"):
            file_path = os.path.join(spell_dir_path, file)
            audio = load_audio_advanced(file_path, augment=True)
            if audio is not None:
                features = extract_advanced_features(audio)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(spell_label)

    # --- Загрузка обычной речи для класса 'unknown' ---
    print(f"\nЗагрузка обычной речи для класса 'unknown' из '{unknown_data_root}'...")
    try:
        # Добавил .wav, так как common voice может содержать и их
        unknown_files = [f for f in os.listdir(unknown_data_root) if f.lower().endswith(('.wav', '.mp3'))]
        random.shuffle(unknown_files)

        # Применяем ограничение в зависимости от режима
        if LIGHT_MODE:
            files_to_load_unknown = unknown_files[:MAX_UNKNOWN_SAMPLES_LIGHT]
        else:
            files_to_load_unknown = unknown_files[:MAX_UNKNOWN_SAMPLES] # Используем общее ограничение

        unknown_label = label_map['unknown']
        for file in tqdm(files_to_load_unknown, desc="Класс 'unknown'"):
            file_path = os.path.join(unknown_data_root, file)
            audio = load_audio_advanced(file_path, augment=True)
            if audio is not None:
                features = extract_advanced_features(audio)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(unknown_label)
    except FileNotFoundError:
        print(f"!!! ОШИБКА: Папка с русской речью не найдена: '{unknown_data_root}'.")
        print("Пожалуйста, убедитесь, что путь верен и папка существует.")

    if not all_features:
        print(
            "\n!!! КРИТИЧЕСКАЯ ОШИБКА: Не загружено ни одного аудиофайла. Проверьте правильность путей и структуру папок.")
        exit()

    print("\n--- Загрузка данных завершена ---")
    label_counts = Counter(all_labels)
    print("Количество сэмплов по классам:")
    for label_idx, count in sorted(label_counts.items()):
        class_name = idx_to_class_name[label_idx]
        print(f"- Класс '{class_name}' (метка {label_idx}): {count} сэмплов")

    # Передача all_features и all_labels в DiskDataset
    dataset = DiskDataset(all_features, all_labels)
    return dataset, label_map, idx_to_class_name


def create_class_weights(all_labels, device):
    # Убедимся, что unique_labels - это список всех возможных меток, чтобы веса вычислялись корректно
    # np.unique(all_labels) вернет только те метки, которые фактически присутствуют в данных.
    # Чтобы создать тензор весов правильного размера для всех классов, используем len(label_map).
    num_total_classes = len(np.unique(all_labels)) # Количество классов, которые реально есть в данных

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),  # Используем только те классы, которые реально есть в данных
        y=np.array(all_labels)
    )
    # Создаем полный тензор весов, чтобы его размер соответствовал num_classes модели
    full_class_weights_tensor = torch.zeros(num_total_classes, dtype=torch.float)
    for i, weight in zip(np.unique(all_labels), class_weights):
        full_class_weights_tensor[i] = weight

    full_class_weights_tensor = full_class_weights_tensor.to(device)
    print("\nРассчитаны веса для классов (для борьбы с дисбалансом):")
    print(full_class_weights_tensor)
    return full_class_weights_tensor


def train(model, train_loader, val_loader, class_weights_tensor, device, num_epochs_to_run):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Удален параметр 'verbose'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    num_epochs = num_epochs_to_run
    best_val_loss = float('inf')

    print("\n--- Начало обучения ---")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # Используем enumerate для получения inputs и labels
        for batch_idx, (inputs, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]  ")):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects.double() / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_spell_model.pth')
            print(f"  -> Val loss decreased. Model saved to 'best_spell_model.pth'")

    print("\n--- Обучение завершено ---")
    print(f"Лучшая модель сохранена в файле 'best_spell_model.pth' с Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nМодель будет обучаться на устройстве: {device}")

    # load_data теперь возвращает idx_to_class_name
    dataset, label_map, idx_to_class_name = load_data()

    # Проверяем, достаточно ли классов для обучения
    if len(label_map) < 2:
        print(
            "\n!!! ОШИБКА: Для обучения необходимо как минимум 2 класса. Проверьте, что есть и заклинания, и 'unknown' речь.")
        # Добавляем явный вызов очистки временных файлов в случае ошибки
        if 'dataset' in locals():
            dataset.cleanup()
        exit() # Прерываем выполнение скрипта

    # Здесь используем все метки из dataset.labels
    all_labels_for_weights = dataset.labels.numpy()
    class_weights_tensor = create_class_weights(all_labels_for_weights, device)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    batch_size = 64
    # Увеличил num_workers для лучшей производительности, если позволяет система
    # Ограничение num_workers в зависимости от количества доступных ядер процессора
    num_workers_to_use = os.cpu_count() // 2 if os.cpu_count() else 4
    if num_workers_to_use == 0: num_workers_to_use = 1 # Гарантируем минимум 1 работник

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers_to_use,
                              pin_memory=True)  # Использование половины ядер CPU
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers_to_use, pin_memory=True)

    print(f"\nДанные разделены: {len(train_set)} для обучения, {len(val_set)} для валидации.")

    model = AdvancedSpellCNN(num_classes=len(label_map)).to(device)

    # Выбираем количество эпох в зависимости от режима
    num_epochs_to_run = NUM_EPOCHS_LIGHT if LIGHT_MODE else 50
    print(f"Будет выполнено {num_epochs_to_run} эпох.")

    # Оборачиваем вызов train в try-finally для гарантированной очистки
    try:
        train(model, train_loader, val_loader, class_weights_tensor, device, num_epochs_to_run)
    finally:
        # Гарантированная очистка временной директории, даже если обучение прервалось ошибкой
        if 'dataset' in locals(): # Проверяем, что объект dataset был создан
            dataset.cleanup()

