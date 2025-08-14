import torch
import os
import soundfile as sf
import re  # Для очистки названий файлов
import random

# Убедитесь, что установлены эти библиотеки:
# pip install torch torchaudio soundfile

# --- Параметры Silero TTS ---
language = 'ru'
model_id = 'v4_ru'
sample_rate = 48000 # v4 модели работают с 48000 Гц для лучшего качества. Модель сама сделает ресемплинг, если нужно.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели Silero TTS
# Это может занять время при первом запуске, так как скачивает модель
try:
    # Загружаем только модель, остальные возвращаемые значения нам не нужны
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id
    )
    model.to(device)  # Перемещаем модель на указанное устройство (CPU)

    # Для v4_ru спикеры предопределены и жестко заданы
    available_speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
    # 'random' позволяет модели самой выбрать голос, что очень удобно для вариативности

    print("Модель Silero TTS успешно загружена!")
    print(f"Доступные спикеры для '{model_id}': {available_speakers}")

except Exception as e:
    print(f"Ошибка при загрузке модели Silero TTS: {e}")
    print("Проверьте подключение к интернету и правильность указанных параметров.")
    exit()

# ==============================================================================
# СЛОВАРЬ ДЛЯ ЗАКЛИНАНИЙ
# Ключ: Правильное название для папки/файла (как в вашем датасете).
# Значение: Текст, который будет произносить TTS (можно использовать '+' для ударений).
# ==============================================================================
spell_pronunciation_map = {
    # Правильное название (ключ)      # Текст для TTS (значение)
    'Акцио':                            'а+кцио',
    'авада_кедавра':                    'ав+ада кед+авра',
    'аресто_моментум':                  'ар+есто мом+энтум',
    'бомбарда':                         'бомб+арда',
    'вингардиум_левиоса':               'вингардиум леви+оса',
    'глациус':                          'гл+ациус',
    'делюминация':                      'делюмин+ация',
    'депульсо':                         'дэп+ульсо',
    'десцендо':                         'дэсц+ендо',
    'диффиндо':                         'дифф+индо',
    'империус':                         'имп+ериус',
    'инсендио':                         'инс+ендио',
    'конфринго':                        'конфр+инго',
    'круцио':                           'кр+уцио',
    'левиосо':                          'леви+осо',
    'люмос':                            'л+юмос',
    'протего':                          'прот+эго',
    'ревелио':                          'рев+елио',
    'репаро':                           'реп+аро',
    'трансформация':                    'трансформ+ация',
    'флиппендо':                        'флипп+ендо',
    'экспеллиармус':                    'экспелли+армус'
}

# --- Функция для безопасного имени файла ---
def sanitize_filename(name):
    """
    Очищает строку для использования в качестве имени файла, заменяя недопустимые символы.
    Пробелы заменяются на подчеркивания.
    """
    return re.sub(r'[\\/:*?"<>| ]', '_', name)

# --- Генерация аудио для каждого заклинания ---
# Новая папка для сгенерированных файлов, чтобы отделить их от Ваших реальных записей
output_base_dir = r"C:\Users\Admin\PycharmProjects\SPELL_NEIRO\audio\raw_tts_data(Silero)"
os.makedirs(output_base_dir, exist_ok=True)

print("\n--- Запуск генерации аудио с помощью Silero TTS ---")
num_variations_per_spell = 10000  # Количество вариаций на каждое заклинание

# Итерация по словарю. Получаем и правильное имя, и текст для озвучки.
for correct_name, tts_text in spell_pronunciation_map.items():
    print(f"\nГенерация для '{correct_name}' (произносим как: '{tts_text}')")

    sanitized_name = sanitize_filename(correct_name)
    spell_dir = os.path.join(output_base_dir, sanitized_name)
    os.makedirs(spell_dir, exist_ok=True)

    for i in range(num_variations_per_spell):
        # Циклично выбираем спикера из доступных
        current_speaker = available_speakers[i % len(available_speakers)]

        try:
            # Модель генерирует аудио на основе текста для TTS (tts_text)
            audio_tensor = model.apply_tts(
                text=tts_text,
                speaker=current_speaker,
                sample_rate=sample_rate
            )

            # Имя файла создается на основе правильного имени заклинания (correct_name)
            filename = f"{sanitized_name}_silero_gen_{i+1}_{current_speaker}.wav"
            output_path = os.path.join(spell_dir, filename)

            sf.write(output_path, audio_tensor.numpy(), sample_rate)
            print(f" - Сохранено: {output_path}")

        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось сгенерировать аудио для '{correct_name}' (вариация {i+1}, спикер {current_speaker}). Ошибка: {e}")

print("\nГенерация с помощью Silero TTS завершена!")
print(f"Новые файлы находятся в: {output_base_dir}")

