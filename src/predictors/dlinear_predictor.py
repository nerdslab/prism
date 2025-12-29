import torch
import numpy as np
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.util import forecast_start
from typing import Iterable, Dict, Any, Optional


class DLinearPredictor(RepresentablePredictor):
    """DLinear predictor for GluonTS evaluation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        prediction_length: int,
        ctx_len: int = 336,
        device: str = "cuda",
    ):
        super().__init__(prediction_length=prediction_length)
        self.model = model.eval().to(device)
        self.ctx_len = ctx_len
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def predict(self, dataset: Iterable[Dict[str, Any]], **kwargs) -> Iterable[SampleForecast]:
        """Generate forecasts for the given dataset."""
        for entry in dataset:
            # Extract target time series
            tgt = np.asarray(entry["target"], dtype=np.float32)
            if tgt.ndim == 1:
                tgt = tgt[:, None]  # [T, 1]
            
            # Build context window
            T, C = tgt.shape
            H = self.prediction_length
            
            # Create context tensor
            ctx = np.zeros((self.ctx_len, C), dtype=np.float32)
            take = max(0, min(self.ctx_len, T - H))
            if take > 0:
                ctx[-take:, :] = tgt[T - H - take:T - H, :]
            
            # Convert to tensor and add batch dimension
            x = torch.from_numpy(ctx).unsqueeze(0).to(self.device)  # [1, L, C]
            
            # Generate forecast
            with torch.no_grad():
                yhat = self.model(x)  # [1, H, C]
                yhat = yhat.squeeze(0).detach().cpu().numpy()  # [H, C]
            
            # Clean any NaN/Inf values
            yhat = np.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)
            yhat = np.clip(yhat, -1e6, 1e6).astype(np.float32)
            
            # Create SampleForecast (DLinear produces deterministic output, so we use single sample)
            samples = yhat[None, ...]  # Add sample dimension: [1, H, C]
            
            yield SampleForecast(
                samples=samples,
                start_date=forecast_start(entry)
            )