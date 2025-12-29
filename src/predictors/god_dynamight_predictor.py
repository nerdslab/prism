import torch
import numpy as np
from gluonts.model.forecast import SampleForecast
from gluonts.itertools import batcher
from tqdm.auto import tqdm
from typing import Iterable, Dict, Any, List, Optional


class GodDynamightPredictor:
    """A minimal predictor wrapper for your `god_dynamight` model that returns GluonTS Forecasts."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        prediction_length: int,
        ctx_len: int = 336,
        device: str = "cuda",
        num_samples: int = 1,
        lead_time: int = 0,
        clip_value: Optional[float] = 1e6,
    ):
        self.pipeline = model.eval().to(device)
        self.prediction_length = int(prediction_length)
        self.lead_time = lead_time
        self.ctx_len = int(ctx_len)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_samples = int(num_samples)
        self.clip_value = clip_value

    @torch.no_grad()
    def predict(
        self,
        test_data_input: Iterable[Dict[str, Any]],
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> List:
        """
        Convert an iterable of GluonTS DataEntry dicts into a list of Forecast objects.
        """
        forecasts = []
        pipeline = self.pipeline

        # Materialize iterable data to avoid empty iterator from multiple iterations
        entries = list(test_data_input)

        # OOM adaptive batch size
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                for batch in tqdm(batcher(entries, batch_size=batch_size), disable=not show_progress):
                    # Build batch tensor [B, L, C]
                    batch_tensors = []
                    for entry in batch:
                        tgt = np.asarray(entry["target"], dtype=np.float32)
                        if tgt.ndim == 1:
                            tgt = tgt[:, None]  # [T, 1]
                        
                        # Build context window
                        L = self.ctx_len
                        H = self.prediction_length
                        ctx = np.zeros((L, tgt.shape[1]), dtype=np.float32)
                        take = max(0, min(L, tgt.shape[0] - H))
                        if take > 0:
                            ctx[-take:, :] = tgt[tgt.shape[0] - H - take:tgt.shape[0] - H, :]
                        # sanitize context
                        ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        batch_tensors.append(torch.from_numpy(ctx))
                    
                    # Stack into batch tensor [B, L, C]
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    
                    # Call model prediction
                    yhat = pipeline.forecast(batch_tensor)  # [B, H, C]
                    if isinstance(yhat, (tuple, list)):
                        yhat = yhat[0]
                    yhat = yhat.detach().cpu().numpy()

                    # Numerical sanitization, avoid NaN/Inf in evaluation
                    yhat = np.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)
                    if not np.isfinite(yhat).all():
                        yhat[~np.isfinite(yhat)] = 0.0
                    yhat = np.clip(yhat, -float(self.clip_value), float(self.clip_value))
                    
                    forecast_outputs.append(yhat)
                forecast_outputs = np.concatenate(forecast_outputs, axis=0)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, entries):
            # Normalize to [S,H,C], here S=1, and sanitize again
            samples = item[None, ...].astype(np.float32)
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.isfinite(samples).all(), "Non-finite values in samples after sanitization"
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(SampleForecast(samples=samples, start_date=forecast_start_date))

        return forecasts
