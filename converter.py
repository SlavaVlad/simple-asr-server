import ffmpeg
import os
import tempfile
import shutil

def is_valid_format(file_path: str) -> bool:
    """Проверяет, является ли аудиофайл 16kHz моно WAV."""
    try:
        probe = ffmpeg.probe(file_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream is None:
            return False

        return (
            audio_stream.get('codec_name') == 'pcm_s16le' and
            audio_stream.get('channels') == 1 and
            audio_stream.get('sample_rate') == '16000'
        )
    except ffmpeg.Error:
        return False

def convert_to_wav(input_file_path: str) -> tuple[str, bool]:
    """
    Конвертирует аудиофайл в 16kHz моно WAV.
    Возвращает путь к сконвертированному файлу и флаг, указывающий, была ли выполнена конвертация.
    Если файл уже в нужном формате, возвращает исходный путь и False.
    """
    if is_valid_format(input_file_path):
        return input_file_path, False

    output_file_path = tempfile.mktemp(suffix=".wav")

    try:
        ffmpeg.input(input_file_path).output(
            output_file_path,
            acodec='pcm_s16le',
            ac=1,
            ar='16k'
        ).run(capture_stdout=True, capture_stderr=True)
        return output_file_path, True
    except ffmpeg.Error as e:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        raise e

