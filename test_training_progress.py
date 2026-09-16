import os
import sys


SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from fizgig.training.progress import (  # noqa: E402
    TrainingProgressTracker,
    parse_epoch_line,
    parse_preview_phase,
    parse_refmod_progress_line,
    parse_tqdm_progress_line,
)


def test_tqdm_parser_uses_the_console_values_verbatim():
    parsed = parse_tqdm_progress_line(
        "minimax-h3: 10%|#| 141/1350 [04:49<41:29, 2.62s/it, avr_loss=0.3441]"
    )
    assert parsed == {
        "step": 141,
        "total_steps": 1350,
        "eta_text": "41:29",
        "speed_text": "2.62 s/it",
        "average_loss_text": "0.3441",
    }


def test_tqdm_parser_accepts_iterations_per_second():
    parsed = parse_tqdm_progress_line(
        "minimax-h3: 1%| | 14/1350 [00:14<20:32, 1.08it/s, avr_loss=0.4337]"
    )
    assert parsed["speed_text"] == "1.08 it/s"


def test_tqdm_parser_rejects_preview_sampling_bar():
    assert parse_tqdm_progress_line(
        "sampling: 38%|###| 3/8 [00:08<00:13, 2.79s/it]"
    ) is None


def test_preview_phase_parser():
    assert parse_preview_phase("INFO: rendering previews (epoch 6) on the training DiT") == ("start", 6)
    assert parse_preview_phase("[preview] epoch 7: prompt 1/1") == ("start", 7)
    assert parse_preview_phase("[preview] epoch 7: wrote 1 sample(s)") == ("complete", 7)
    assert parse_preview_phase("[preview] epoch 7 preview failed (OSError)") == ("failed", 7)
    assert parse_preview_phase("minimax-h3: 141/1350") is None


def test_epoch_parser_accepts_all_trainer_styles():
    assert parse_epoch_line("epoch 6/50") == (6, 50)
    assert parse_epoch_line("INFO:fizgig:epoch 6/50 done — avr_loss 0.3") == (6, 50)
    assert parse_epoch_line("[resume] continuing at epoch 7/50 (global_step 162)") == (7, 50)


def test_tracker_infers_epoch_without_trainer_instrumentation():
    tracker = TrainingProgressTracker(total_epochs=50)
    update = tracker.consume(
        "minimax-h3: 10%|#| 141/1350 [04:49<41:29, 2.62s/it, avr_loss=0.3441]"
    )
    assert update == {
        "kind": "training",
        "epoch": 6,
        "total_epochs": 50,
        "step": 141,
        "total_steps": 1350,
        "eta_text": "41:29",
        "speed_text": "2.62 s/it",
        "average_loss_text": "0.3441",
    }


def test_tracker_keeps_preview_sampling_separate():
    tracker = TrainingProgressTracker(total_epochs=30)
    assert tracker.consume("INFO: rendering previews (epoch 2) on the training DiT") == {
        "kind": "preview", "phase": "start", "epoch": 2,
    }
    assert tracker.consume("sampling: 38%|###| 3/8 [00:08<00:13, 2.79s/it]") is None


def test_refmod_parser_uses_its_console_values_and_calculates_eta():
    parsed = parse_refmod_progress_line(
        "[refmod] step 40/200  loss 0.1234  drift 0.042 (rms, latent units)  1.50 s/step"
    )
    assert parsed == {
        "step": 40,
        "total_steps": 200,
        "eta_text": "04:00",
        "speed_text": "1.50 s/step",
        "average_loss_text": "0.1234",
    }


def test_tracker_marks_refmod_as_a_step_only_run():
    tracker = TrainingProgressTracker(total_epochs=50)
    update = tracker.consume(
        "[refmod] step 10/200  loss 0.4321  drift 0.010 (rms, latent units)  2.00 s/step"
    )
    assert update["kind"] == "refmod"
    assert update["step"] == 10
    assert update["total_steps"] == 200
    assert "epoch" not in update
