# Canary-Qwen-2.5B

[![Replicate](https://replicate.com/zsxkib/canary-qwen-2.5b/badge)](https://replicate.com/zsxkib/canary-qwen-2.5b)

You know how most speech AI either just transcribes what you say or makes you choose between speech-to-text OR smart analysis? Canary-Qwen does both. Transcribe a 2-hour meeting and get perfect text with timestamps. Then ask it to summarize the key points, extract action items, or answer questions about what was discussed.

This is NVIDIA's Canary speech tech married to Qwen's language model. It doesn't just hear - it understands.

## Try it right now

Got an NVIDIA GPU and Docker? Three commands and you're processing audio:

```bash
git clone https://github.com/zsxkib/cog-nvidia-canary-qwen-2.5b
cd cog-nvidia-canary-qwen-2.5b
cog predict -i audio=@meeting.wav
```

That's it. No setup, no hunting for model weights. It downloads everything and starts working.

## What makes this different

Most speech models do one thing. Canary-Qwen does two things really well.

First, it transcribes audio. You can throw 2 hours of audio at it and it'll write down what everyone said with punctuation and capitalization. It runs at 418x real-time speed - a 1-minute recording processes in 0.14 seconds.

Second, it analyzes what's happening in the audio. Ask it questions about a podcast transcript and it can tell you what the host was discussing. Feed it a meeting recording and it can summarize key points, extract action items, or answer specific questions about the content.

The speech recognition achieves 5.63% word error rate on standard benchmarks - that's state-of-the-art territory.

## Some things you can try

```bash
# Basic transcription with timestamps
cog predict -i audio=@interview.mp3 -i include_timestamps=true

# Ask questions about the audio content
cog predict -i audio=@lecture.wav -i llm_prompt="Summarize the main points from this lecture in bullet points"

# Extract specific information
cog predict -i audio=@meeting.wav -i llm_prompt="What are the action items from this meeting?"

# See the model's reasoning process
cog predict -i audio=@discussion.wav -i llm_prompt="Who are the main speakers and what do they disagree about?" -i show_confidence=true

# Turn off timestamps for cleaner text
cog predict -i audio=@podcast.mp3 -i include_timestamps=false

# Process a long recording
cog predict -i audio=@conference_call.wav -i llm_prompt="Create a summary with speaker names and key decisions made"
```

## All the parameters

- `audio` - Your audio file (supports MP3, WAV, M4A, FLAC, OGG, AAC - up to 2 hours)
- `llm_prompt` - Question or instruction for analyzing the transcript (optional)
- `include_timestamps` - Add time markers to transcript (default: true)  
- `show_confidence` - Include the model's reasoning process (default: false)

## What you need

- NVIDIA GPU with CUDA support
- Docker 
- Cog (see https://cog.run)

## Use cases

Content creators transcribing and analyzing podcasts, interviews, YouTube videos. Businesses processing meeting recordings and extracting insights. Researchers working with recorded interviews. Students turning lecture recordings into organized notes.

Good when you need both accurate transcription AND smart analysis of what was actually said.

## How it works

NVIDIA took their Canary speech recognition model and combined it with Qwen's language model. The speech encoder processes audio in 40-second chunks, then the language model can work with the resulting transcript to answer questions, summarize, or analyze the content.

It's a hybrid approach - dedicated speech recognition for accuracy, then language model capabilities for understanding.

## Performance 

Canary-Qwen achieves a 5.63% average word error rate across standard benchmarks, placing it among the top-performing speech recognition systems available. But benchmarks are just numbers - try it on your audio and see how it handles real-world messiness.

## What's included

- `predict.py` - Main Cog interface with transcription and analysis modes
- `cog.yaml` - Cog configuration  
- Model weights download automatically from Hugging Face

## Languages supported

English only. While the underlying components were trained on some other languages, this model is optimized specifically for English speech.

## Limitations

- English only
- 2 hours max per file
- Works best with clear conversational speech
- Analysis prompts work best with transcripts under 1000 words

## About the model

Canary-Qwen-2.5B was created by NVIDIA and released under a CC-BY-4.0 license. It combines NVIDIA's Canary-1B speech recognition with Qwen3-1.7B language model using 2.5 billion parameters total.

Trained on 234,000 hours of English speech from conversations, web videos, and audiobooks.

---

**Follow me:**
- Twitter: [@zsxkib](https://twitter.com/zsxkib)
- GitHub: [@zsxkib](https://github.com/zsxkib)
- Replicate: [@zsxkib](https://replicate.com/zsxkib)
