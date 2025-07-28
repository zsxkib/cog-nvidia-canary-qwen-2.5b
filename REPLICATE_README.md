---
name: canary-qwen-2.5b
description: Turn speech into text, then analyze it with AI - transcribe meetings and get smart insights about what was discussed 🎙️🧠
github_url: https://github.com/zsxkib/cog-nvidia-canary-qwen-2.5b
weights_url: https://huggingface.co/nvidia/canary-qwen-2.5b
paper_url: https://arxiv.org/abs/2407.10963
license_url: https://huggingface.co/nvidia/canary-qwen-2.5b/blob/main/LICENSE
---

# Canary-Qwen-2.5B: Speech Recognition + AI Analysis 🎙️🧠

You know how most speech AI either just writes down what you said or completely misses the context? Canary-Qwen-2.5B actually gets both. Upload a 2-hour meeting recording and get perfect transcription with timestamps. Then ask it "What were the main decisions made?" and it analyzes the content like a human would—understanding context, extracting insights, and reasoning through complex discussions step by step.

> **Note:**  
> The model weights are **for commercial and non-commercial use** under NVIDIA's CC-BY-4.0 license.  
> **Attribution is required when using the model.**

## What Canary-Qwen-2.5B does ✨

Canary-Qwen-2.5B transforms how you analyze audio by:
- **Perfect transcription**: Goes beyond basic speech-to-text to understand punctuation, capitalization, and conversation flow
- **Smart reasoning**: Shows its thinking process when analyzing complex discussions or presentations  
- **Dual capability**: Handles pure transcription and intelligent analysis equally well
- **Long-form processing**: Analyzes up to 2 hours of audio content in context
- **Question answering**: Answer any question about your audio content
- **Professional quality**: Provides detailed analysis suitable for business and research

## Audio analysis capabilities 🎵

Canary-Qwen-2.5B uses advanced reasoning to understand not just what words are spoken, but their relationships, context, and meaning. It can analyze everything from casual conversations to formal presentations.

Key features:

🧠 **Contextual understanding** for detailed conversation analysis  
🎙️ **Speaker awareness** with context and discussion flow recognition  
🎼 **Content structure** including topics, decisions, and action items  
🔊 **Audio quality handling** for real-world recordings  
📝 **Flexible questioning** - ask it anything about your audio  
⏱️ **Extended processing** up to 2 hours of continuous audio  
🎯 **Timestamp precision** for focusing on specific segments

## How to get the best results 🌟

**Basic approach:**
- Canary-Qwen works with MP3, WAV, M4A, FLAC, OGG, AAC, anything
- Ask clear questions about what you want to know  
- Get detailed insights and analysis

**Advanced control:**
- Use specific prompts to customize the analysis format
- Enable reasoning mode for step-by-step thinking
- Include timestamps for detailed segment analysis
- Use targeted questions for focused insights

**Example questions:**

*For business meetings:*
- "What were the main action items and who is responsible for each?"
- "Summarize the key decisions made and any concerns raised"
- "Who were the speakers and what were their main points?"

*For interviews and podcasts:*
- "What are the most interesting insights shared by the guest?"
- "Create bullet points of the main topics discussed"  
- "What questions did the host ask and how did the guest respond?"

*For lectures and presentations:*
- "Break down the main concepts explained in this lecture"
- "What examples were used to illustrate key points?"
- "Create a structured summary with headings and key takeaways"

## Parameter controls 🎛️

**include_timestamps** (true/false): Adds time markers to transcription. When enabled, shows exactly when each segment was spoken.

**show_confidence** (true/false): Reveals the model's reasoning process. Shows how it arrived at its analysis and conclusions.

**llm_prompt**: Your question or instruction for analysis. Be specific about what insights you want.

## What makes Canary-Qwen-2.5B special 🚀

Traditional speech models are limited to basic transcription or simple commands. Canary-Qwen-2.5B changes this by:

- **True dual capability**: Combines NVIDIA's state-of-the-art speech recognition with Qwen's language understanding in one seamless model
- **Advanced reasoning**: Doesn't just transcribe—analyzes content, understands context, and answers complex questions
- **Unified approach**: No need to use separate tools for transcription and analysis  
- **Professional interface**: Works seamlessly through Replicate's web interface
- **Enterprise quality**: 5.63% word error rate puts it among the best speech recognition systems available

## Best use cases 🎯

Canary-Qwen-2.5B excels at:
- **Meeting analysis**: Extract action items, decisions, and key discussions from recordings
- **Content creation**: Transcribe interviews, podcasts, and videos for editing and repurposing
- **Research processing**: Analyze recorded interviews, focus groups, and qualitative data  
- **Educational support**: Convert lectures and presentations into structured notes and summaries
- **Business intelligence**: Process customer calls and feedback for insights and trends
- **Accessibility services**: Create accurate transcripts with intelligent summaries for hearing-impaired users

## Limitations to consider ⚠️

- Works best with clear English speech
- Background noise may affect transcription accuracy
- Complex audio scenarios benefit from specific questioning  
- Processing time increases with longer audio files
- Analysis quality depends on audio clarity and content complexity

## Research background 📚

Canary-Qwen-2.5B represents a significant advance in speech-language models, built on NVIDIA's research in speech recognition and Alibaba's work in language understanding. It combines FastConformer speech encoding with Transformer-based language modeling.

The model builds on the Canary series, introducing:
- Hybrid speech-language architecture
- 2.5 billion parameter optimization
- Multi-modal reasoning capabilities  
- Extended context processing

Original research: [Less is More: Accurate Speech Recognition & Translation without Web-Scale Data](https://arxiv.org/abs/2407.10963)

## Important licensing note 📝

> **This model is for commercial and non-commercial use** under NVIDIA's CC-BY-4.0 license.  
> **Attribution is required** when using the model.

The model is built on NVIDIA Canary-1B and Qwen3-1.7B which have their own open-source license requirements.

Built by NVIDIA. Packaged for Replicate by [@zsxkib](https://github.com/zsxkib).

## Disclaimer ‼️

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.

---

⭐ Star the repo on [GitHub](https://github.com/zsxkib/cog-nvidia-canary-qwen-2.5b)!  
🐦 Follow [@zsxkib](https://twitter.com/zsxkib) on X  
💻 Check out more projects [@zsxkib](https://github.com/zsxkib) on GitHub 