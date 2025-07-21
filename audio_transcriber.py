from pydub import AudioSegment
from openai import AzureOpenAI
from openai import OpenAI
import os
import re
from datetime import datetime, timedelta

def adjust_timestamps(file_path, output_path, time_delta_minutes=10):
    time_format = "%H:%M:%S,%f"
    time_delta = timedelta(minutes=time_delta_minutes)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(output_path, 'w') as output_file:
        for line in lines:
            timestamps = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', line)
            for timestamp in timestamps:
                time_obj = datetime.strptime(timestamp, time_format)
                new_time = time_obj + time_delta
                new_timestamp = new_time.strftime(time_format)[:-3]
                line = line.replace(timestamp, new_timestamp)
            output_file.write(line)


def get_modified_transcript(transcript, i):
    transcript_file = f'transcript{i + 1}.srt'
    with open(transcript_file, 'w') as file:
        file.write(transcript)
    adjusted_transcript_file = f'adjusted_transcript{i + 1}.srt'
    adjust_timestamps(transcript_file, adjusted_transcript_file, time_delta_minutes=10 * (i))
    with open(adjusted_transcript_file, 'r') as file:
        mod_transcript = file.read()
    return mod_transcript


class WhisperAudioTranscriber:
    def __init__(self, openai_api_key: str) -> None:
        self.api_key = openai_api_key
        # self.api_endpoint = api_endpoint
        self.total_chunks = 0
        self.chunk_size = 0
        self.audio = None
        self.ext = None
        self.chunks_filename = []

    def load_file(self, file_path: str, chunk_size_in_min: int = 10):
        # Load input audio file
        if '.' in file_path:
            self.ext = file_path.split('.')[-1]
        else:
            raise "File does not have extention type"
        try:
            # file_path = "./DownloadedVideos/1ApHrW6bsTPWG1wR8gi4a4Og78Czphkpw_240_1700.mp3"
            self.audio = AudioSegment.from_mp3(file_path)
        except Exception as e:
            raise e
        # Define the chunk size in milliseconds (default 10 minutes)
        chunk_size = chunk_size_in_min * 60 * 1000
        self.chunk_size = chunk_size
        # Calculate the total number of chunks
        total_chunks = len(self.audio) // chunk_size + 1
        self.total_chunks = total_chunks
        print(total_chunks)
        return self.audio

    def create_audio_chunks(self, chunk_file_name: str = 'chunk') -> bool:
        if not self.total_chunks or not self.chunk_size or not self.audio or not self.ext:
            raise "Please load_file first to create chunks"
        try:
            # Split the audio into chunks and export them
            for i in range(self.total_chunks):
                start_time = i * self.chunk_size
                end_time = (i + 1) * self.chunk_size
                chunk = self.audio[start_time:end_time]

                # Export each chunk with a unique filename
                file_name = f"{chunk_file_name}_{i + 1}.{self.ext}"
                chunk.export(file_name, format=self.ext)
                self.chunks_filename.append(file_name)

        except Exception as e:
            print(e)
            return False
        return True

    def start_transcribing(self, output_filename='output_transcript') -> str:
        if len(self.chunks_filename) == 0:
            raise "Please create_audio_chunks first to start transcribing"
        try:
            output_file_name = f"{output_filename}"
            with open(output_file_name, 'w', encoding="utf-8") as file:
                for i in range(self.total_chunks):
                    chunk_file_name = self.chunks_filename[i]
                    audio_file = open(chunk_file_name, "rb")
                    client = OpenAI(
                        api_key=self.api_key
                    )
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="srt",
                        language="en",
                        prompt="The following is a technical interview conducted online. Remove punctuations and make them subtle. Don't add anything new"
                    )
                    mod_transcript = get_modified_transcript(transcript, i)
                    file.write(str(mod_transcript))
        except Exception as e:
            print(e)
            return ""
        return output_file_name