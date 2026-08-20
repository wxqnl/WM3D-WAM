from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from wm3d_wam.data import online_episode
from wm3d_wam.data.online_episode import OnlineEpisodeError


def test_corrupt_pyav_bitstream_becomes_a_retryable_online_error(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise av.InvalidDataError(1094995529, "invalid bitstream")

    monkeypatch.setattr(online_episode, "_decode_video_rows_impl", fail)
    with pytest.raises(OnlineEpisodeError, match="PyAV failed to decode"):
        online_episode._decode_video_rows(
            Path("broken.mp4"),
            start_s=0.0,
            stop_s=1.0,
            observation_times_s=np.asarray([0.0, 0.1]),
            rows=np.asarray([0]),
        )

