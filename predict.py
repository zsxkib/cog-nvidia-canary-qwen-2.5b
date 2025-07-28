import os
# Set environment variables for model caching
MODEL_CACHE = "model_cache"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import tempfile
import uuid
from datetime import timedelta
from typing import Optional
import torch
import soundfile as sf
import torchaudio.transforms as T
from cog import BasePredictor, Input, Path
from lhotse import Recording
from lhotse.dataset import DynamicCutSampler
from nemo.collections.speechlm2 import SALM

# FSDP patch for compatibility
try:
    from torch.distributed import fsdp
    if not hasattr(fsdp, 'fully_shard'):
        fsdp.fully_shard = lambda *a, **k: a[0] if len(a) == 1 and callable(a[0]) else lambda f: f
except ImportError:
    pass


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ.update({
            "HF_HOME": MODEL_CACHE,
            "TORCH_HOME": MODEL_CACHE, 
            "HF_DATASETS_CACHE": MODEL_CACHE,
            "TRANSFORMERS_CACHE": MODEL_CACHE,
            "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCH_CUDNN_V8_API_ENABLED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        })
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        self.model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().eval().to(self.device)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        llm_prompt: Optional[str] = Input(description="Optional LLM analysis prompt", default=None),
        include_timestamps: bool = Input(description="Include timestamps in transcript", default=True),
        show_confidence: bool = Input(description="Show AI reasoning in analysis", default=False),
    ) -> str:
        audio_path = str(audio)
        if os.path.splitext(audio_path)[1].lower() not in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"):
            raise ValueError("Unsupported format")
        
        # Load and prepare audio
        recording_id = str(uuid.uuid4())
        audio_data, original_sr = sf.read(audio_path)
        
        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample to 16kHz if needed
        if original_sr != 16000:
            resampler = T.Resample(original_sr, 16000)
            audio_data = resampler(torch.from_numpy(audio_data).float()).numpy()
        
        # Create temporary file and recording
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, 16000)
            recording = Recording.from_file(tmp_file.name, recording_id=recording_id)
            tmp_path = tmp_file.name
        
        if recording.duration / 60.0 > 120:
            os.unlink(tmp_path)
            raise ValueError(f"Audio too long: {recording.duration / 60.0:.1f} min")
        
        cut = recording.to_cut()
        audio_batches = DynamicCutSampler(cut.cut_into_windows(40.0), max_cuts=96)
        
        # Process audio in batches
        transcripts = []
        timestamped_transcripts = []
        overall_chunk_idx = 0
        
        for batch in audio_batches:
            audio_data, audio_lengths = batch.load_audio(collate=True)
            with torch.inference_mode():
                prompts = [[{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}"}]] * len(batch)
                output_ids = self.model.generate(
                    prompts=prompts,
                    audios=torch.as_tensor(audio_data).to(self.device, non_blocking=True),
                    audio_lens=torch.as_tensor(audio_lengths).to(self.device, non_blocking=True),
                    max_new_tokens=256,
                )
            
            chunk_transcripts = [self.model.tokenizer.ids_to_text(output_ids_chunk) for output_ids_chunk in output_ids.cpu()]
            for transcript in chunk_transcripts:
                if include_timestamps:
                    start = timedelta(seconds=overall_chunk_idx * 40.0)
                    end = timedelta(seconds=(overall_chunk_idx + 1) * 40.0)
                    timestamped_transcripts.append(f"[{start} - {end}] {transcript}\n\n")
                transcripts.append(transcript)
                overall_chunk_idx += 1
        
        # Cleanup
        os.unlink(tmp_path)
        
        raw_transcript = " ".join(transcripts)
        
        # LLM analysis if requested
        if llm_prompt:
            with torch.inference_mode():
                prompts = [[{"role": "user", "content": f"{llm_prompt}\n\n{raw_transcript}"}]]
                if hasattr(self.model, 'llm') and hasattr(self.model.llm, 'disable_adapter'):
                    with self.model.llm.disable_adapter():
                        output_ids = self.model.generate(prompts=prompts, max_new_tokens=2048)
                else:
                    output_ids = self.model.generate(prompts=prompts, max_new_tokens=2048)
            
            response = self.model.tokenizer.ids_to_text(output_ids[0].cpu())
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
            
            thinking_process = ""
            if "<think>" in response:
                response = response.split("<think>")[-1]
                if "</think>" in response:
                    thinking_process, response = response.split("</think>", 1)
            
            if show_confidence and thinking_process:
                return f"{response.strip()}\n\n---\nReasoning: {thinking_process.strip()}"
            return response.strip()
        
        if include_timestamps:
            return "".join(timestamped_transcripts).strip()
        return raw_transcript.strip()






