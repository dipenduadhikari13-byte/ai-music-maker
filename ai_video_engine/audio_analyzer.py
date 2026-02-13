"""
Audio Analyzer — extract musical structure, tempo, beats, sections, mood
from an audio file using librosa.

Used by the pipeline to inform scene planning: section boundaries tell
the scene planner *where* to cut, beat times drive transition sync,
and energy/mood shape the visual style of each scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SongSection:
    """One continuous section of the song (intro, verse, chorus …)."""
    label: str
    start_time: float
    end_time: float
    duration: float
    energy: float          # 0‥1 normalised average energy


@dataclass
class AudioAnalysis:
    """Complete analysis of a song."""
    path: str
    duration: float
    sample_rate: int
    tempo: float
    beat_times: List[float]
    sections: List[SongSection]
    energy_curve: np.ndarray    # per‐frame RMS (normalised 0‥1)
    energy_times: np.ndarray    # time axis matching energy_curve
    mood: str                   # energetic | mellow | dark | upbeat | dramatic
    key: Optional[str] = None
    rms_db: float = 0.0

    def section_at(self, t: float) -> Optional[SongSection]:
        """Return the section active at time *t*."""
        for s in self.sections:
            if s.start_time <= t < s.end_time:
                return s
        return self.sections[-1] if self.sections else None


class AudioAnalyzer:
    """Extract musical features from an audio file."""

    def analyze(self, audio_path: str) -> AudioAnalysis:
        import librosa

        logger.info("Loading audio: %s", audio_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        logger.info("Duration %.1fs  SR %d", duration, sr)

        # ── tempo & beats ──────────────────────────────────────────
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo_arr)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        logger.info("Tempo %.1f BPM  Beats %d", tempo, len(beat_times))

        # ── energy curve (RMS) ─────────────────────────────────────
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_max = rms.max() if rms.max() > 0 else 1.0
        rms_norm = rms / rms_max
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

        rms_db = float(20 * np.log10(rms.mean() + 1e-8))

        # ── section detection ──────────────────────────────────────
        sections = self._detect_sections(y, sr, duration, rms_norm, rms_times)
        logger.info("Sections detected: %d", len(sections))

        # ── mood estimation ────────────────────────────────────────
        mood = self._estimate_mood(y, sr, tempo, rms, duration)
        logger.info("Estimated mood: %s", mood)

        # ── key detection ──────────────────────────────────────────
        key = self._detect_key(y, sr)
        logger.info("Detected key: %s", key)

        return AudioAnalysis(
            path=audio_path,
            duration=duration,
            sample_rate=sr,
            tempo=tempo,
            beat_times=beat_times,
            sections=sections,
            energy_curve=rms_norm,
            energy_times=rms_times,
            mood=mood,
            key=key,
            rms_db=rms_db,
        )

    # ── section detection ──────────────────────────────────────────────

    def _detect_sections(
        self,
        y: np.ndarray,
        sr: int,
        duration: float,
        rms_norm: np.ndarray,
        rms_times: np.ndarray,
    ) -> List[SongSection]:
        """
        Detect structural sections using spectral novelty.

        Falls back to uniform splitting if librosa.segment is not
        precise enough for the given track.
        """
        import librosa

        sections: List[SongSection] = []

        try:
            # Compute a Mel spectrogram and derive a recurrence / novelty curve
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
            S_db = librosa.power_to_db(S, ref=np.max)

            # Novelty from spectral flux
            novelty = np.sqrt(np.maximum(0, np.diff(S_db, axis=1)).sum(axis=0))
            novelty = novelty / (novelty.max() + 1e-8)

            # Pick peaks in the novelty curve as section boundaries
            from scipy.signal import find_peaks
            peak_distance = int(sr / 512 * 8)  # ~8 seconds min between boundaries
            peaks, props = find_peaks(novelty, height=0.25, distance=peak_distance)
            boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512).tolist()

            # Ensure first and last boundaries
            if not boundary_times or boundary_times[0] > 2.0:
                boundary_times.insert(0, 0.0)
            if boundary_times[-1] < duration - 2.0:
                boundary_times.append(duration)

            # Remove boundaries too close together (<4 s)
            cleaned = [boundary_times[0]]
            for bt in boundary_times[1:]:
                if bt - cleaned[-1] >= 4.0:
                    cleaned.append(bt)
            if cleaned[-1] < duration - 1.0:
                cleaned.append(duration)
            boundary_times = cleaned

        except Exception as exc:
            logger.warning("Section novelty detection failed (%s), using uniform split", exc)
            boundary_times = self._uniform_boundaries(duration)

        # If too few sections, fall back
        if len(boundary_times) < 3:
            boundary_times = self._uniform_boundaries(duration)

        # Label sections heuristically
        labels = self._label_sections(boundary_times, rms_norm, rms_times, duration)

        for i in range(len(boundary_times) - 1):
            start = boundary_times[i]
            end = boundary_times[i + 1]

            # average energy in this section
            mask = (rms_times >= start) & (rms_times < end)
            energy = float(rms_norm[mask].mean()) if mask.any() else 0.5

            sections.append(SongSection(
                label=labels[i],
                start_time=round(start, 2),
                end_time=round(end, 2),
                duration=round(end - start, 2),
                energy=round(energy, 3),
            ))

        return sections

    def _uniform_boundaries(self, duration: float) -> List[float]:
        """Split the song into roughly equal chunks of ~15–25 s."""
        num = max(4, int(duration / 20))
        step = duration / num
        return [round(i * step, 2) for i in range(num + 1)]

    def _label_sections(
        self,
        boundaries: List[float],
        rms_norm: np.ndarray,
        rms_times: np.ndarray,
        duration: float,
    ) -> List[str]:
        """Assign intro/verse/chorus/bridge/outro labels based on energy pattern."""
        n = len(boundaries) - 1
        if n <= 0:
            return ["verse"]

        energies: List[float] = []
        for i in range(n):
            mask = (rms_times >= boundaries[i]) & (rms_times < boundaries[i + 1])
            energies.append(float(rms_norm[mask].mean()) if mask.any() else 0.5)

        median_e = float(np.median(energies))
        labels: List[str] = []

        for i, e in enumerate(energies):
            if i == 0 and e < median_e * 0.8:
                labels.append("intro")
            elif i == n - 1 and e < median_e * 0.8:
                labels.append("outro")
            elif e >= median_e * 1.15:
                labels.append("chorus")
            elif e < median_e * 0.75:
                labels.append("bridge")
            else:
                labels.append("verse")

        return labels

    # ── mood estimation ────────────────────────────────────────────────

    def _estimate_mood(
        self,
        y: np.ndarray,
        sr: int,
        tempo: float,
        rms: np.ndarray,
        duration: float,
    ) -> str:
        import librosa

        # spectral centroid → brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = float(centroid.mean() / sr)  # normalised

        mean_rms = float(rms.mean())
        rms_std = float(rms.std())

        if tempo > 130 and mean_rms > 0.15 and brightness > 0.12:
            return "energetic"
        if tempo > 110 and brightness > 0.1:
            return "upbeat"
        if mean_rms < 0.08 and tempo < 100:
            return "mellow"
        if brightness < 0.08 and mean_rms > 0.1:
            return "dark"
        if rms_std > 0.1:
            return "dramatic"
        return "upbeat"

    # ── key detection ──────────────────────────────────────────────────

    def _detect_key(self, y: np.ndarray, sr: int) -> Optional[str]:
        import librosa

        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_avg = chroma.mean(axis=1)

            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key_idx = int(np.argmax(chroma_avg))

            # major / minor heuristic via the 3rd interval
            major_third = chroma_avg[(key_idx + 4) % 12]
            minor_third = chroma_avg[(key_idx + 3) % 12]
            mode = "major" if major_third >= minor_third else "minor"

            return f"{keys[key_idx]} {mode}"
        except Exception:
            return None
