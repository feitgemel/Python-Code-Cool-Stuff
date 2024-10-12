### Import the `WhisperModel` from the `faster_whisper` library, which is used for transcribing audio files. Also, import `Translator` from the `googletrans` library for text translation capabilities.
from faster_whisper import WhisperModel  
from googletrans import Translator

### Specify the size of the Whisper model to use. "large-v3" is a pre-trained model that provides high accuracy.
model_size = "large-v3"

### Instantiate the Whisper model, setting it to use a GPU for faster processing. The `compute_type="float16"` means using half-precision for computations, which speeds up processing while maintaining accuracy.
model = WhisperModel(model_size, device="cuda", compute_type="float16")

### Define the path to the audio file you want to transcribe. In this case, the file is "StarTrek-Origin.m4v" located in the specified folder.
starFile = "Python-Code-Cool-Stuff/Fast-Whisper/StarTrek-Origin.m4v"

### Perform the transcription of the audio file using beam search with a beam size of 5 for better accuracy. The output is a generator for transcription segments and additional information about the detected language.
segments, info = model.transcribe(starFile, beam_size=5)  
### Convert the generator output to a list for easier manipulation and iteration.
segments = list(segments)

### Print out the detected language and the probability associated with this detection, giving an indication of how confident the model is about the language recognition.
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

### Define a helper function to format time in seconds into the SRT subtitle format (hh:mm:ss,ms). This function is used to create time stamps for subtitles.
def format_timestamp(seconds):  
    hours = int(seconds // 3600)  
    minutes = int((seconds % 3600) // 60)  
    seconds = int(seconds % 60)  
    milliseconds = int((seconds % 1) * 1000)  
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

### Create an English subtitle file in the SRT format. Open a new file for writing, using UTF-8 encoding to support various characters.
with open("Python-Code-Cool-Stuff/Fast-Whisper/StarTrek-Origin.srt", "w", encoding="utf-8") as srt_file:  
    ### Iterate through the transcribed segments, numbering each subtitle block.
    for i, segment in enumerate(segments, start=1):  
        ### Format the start and end times of each segment using the `format_timestamp` function.
        start_time = format_timestamp(segment.start)  
        end_time = format_timestamp(segment.end)  
        text = segment.text  

        ### Print the segment details, showing the start time, end time, and the actual transcribed text.
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        ### Write the formatted subtitle block to the SRT file. Each block includes the index number, time range, and the text content.
        srt_file.write(f"{i}\n")  
        srt_file.write(f"{start_time} --> {end_time}\n")  
        srt_file.write(f"{text}\n\n")

### Confirm the successful creation of the English subtitle file by printing a message.
print("English SRT file generated successfully.")  
print("*******************************")

### The following section will generate French subtitles using the `googletrans` library for translation. Make sure to install googletrans if you haven't already by running: `pip install googletrans==4.0.0-rc1`

### Initialize the `Translator` object to use Google's translation services.
translator = Translator()

### Open a new SRT file to store the translated French subtitles.
with open("Python-Code-Cool-Stuff/Fast-Whisper/StarTrek-Origin-French.srt", "w", encoding="utf-8") as srt_file_es:  
    ### Iterate through the segments, translating each English subtitle to French.
    for i, segment in enumerate(segments, start=1):  
        ### Format the start and end times of each segment just like before.
        start_time = format_timestamp(segment.start)  
        end_time = format_timestamp(segment.end)  
        text = segment.text  

        ### Translate the segment text from English to French using the `translator` object. Specify 'en' as the source language and 'fr' as the destination.
        translated_text = translator.translate(text, src='en', dest='fr').text  
        ### Print the translated segment details.
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, translated_text))

        ### Write the translated subtitle block to the French SRT file, following the SRT format.
        srt_file_es.write(f"{i}\n")  
        srt_file_es.write(f"{start_time} --> {end_time}\n")  
        srt_file_es.write(f"{translated_text}\n\n")

### Print a success message to indicate the French subtitle file has been generated.
print("French SRT file generated successfully.")  
print("*******************************")
