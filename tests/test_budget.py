"""Token-budget tests: no network, no models."""
import pytest
from PIL import Image

from researchiq.config import settings
from researchiq.llm import budget
from researchiq.llm.rate_limiter import RateLimiter
from researchiq.schemas import VisualElement


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _limiter(clock):
    return RateLimiter(8000, 30, 200_000, clock=clock.now, sleep=clock.sleep)


def test_fit_image_respects_token_cap():
    img = budget.fit_image(Image.new("RGB", (4000, 3000)))
    assert budget.image_tokens(*img.size) <= settings.vlm_max_image_tokens


def test_small_image_untouched():
    img = Image.new("RGB", (200, 200))
    assert budget.fit_image(img).size == (200, 200)


def test_rolling_window_never_exceeds_cap():
    clock = FakeClock()
    lim = _limiter(clock)
    window: list[tuple[float, int]] = []
    for _ in range(40):
        lim.acquire(3000)
        window.append((clock.t, 3000))
        in_last_minute = sum(n for t, n in window if clock.t - t < 60)
        assert in_last_minute <= 7200
    assert clock.t > 0  # it had to wait


def test_blocks_until_oldest_expires():
    clock = FakeClock()
    lim = _limiter(clock)
    lim.acquire(3600)
    lim.acquire(3600)
    t0 = clock.t
    lim.acquire(3600)  # 7200 used -> must wait for the first call to age out
    assert clock.t - t0 >= 59


def test_settle_replaces_estimate_with_actual():
    clock = FakeClock()
    lim = _limiter(clock)
    h = lim.acquire(3000)
    lim.settle(h, 1000)
    assert lim.used() == 1000
    assert lim.run_tokens == 1000


def test_impossible_request_is_refused():
    with pytest.raises(ValueError):
        _limiter(FakeClock()).acquire(9000)


def test_vlm_request_fits_cap_and_drops_text_first(tmp_path):
    from researchiq.ingestion.vlm import build_request

    p = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "white").save(p)
    el = VisualElement(paper_id="p", kind="figure", label="Figure 1", caption="c" * 5000,
                       page=1, nearby_text="n" * 20000, image_path=str(p))
    prompt, b64, est = build_request(el)
    assert est <= settings.per_request_token_cap
    assert est <= int(settings.groq_tpm_limit * 0.9)
    assert b64


def test_key_pool_rotates_on_cooldown_and_spreads_load():
    from researchiq.llm.rate_limiter import KeyPool
    clock = FakeClock()
    pool = KeyPool(["a", "b", "c"], 8000, 30, 200_000, clock=clock.now, sleep=clock.sleep)
    first = [pool.acquire(1000)[0] for _ in range(3)]
    assert sorted(first) == [0, 1, 2]           # round-robin
    pool.cooldown(0, 30)
    picks = {pool.acquire(1000)[0] for _ in range(4)}
    assert 0 not in picks                        # cooling key is skipped
    clock.t += 31
    assert 0 in {pool.acquire(1000)[0] for _ in range(3)}


def test_key_pool_capacity_scales_with_keys():
    from researchiq.llm.rate_limiter import KeyPool
    clock = FakeClock()
    pool = KeyPool(["a", "b"], 8000, 30, 200_000, clock=clock.now, sleep=clock.sleep)
    for _ in range(4):                           # 4 x 3000 = 12k tokens, > one key's 7.2k cap
        pool.acquire(3000)
    assert clock.t == 0                          # no waiting needed with two keys


def test_settings_merge_and_dedupe_keys(tmp_path, monkeypatch):
    from researchiq.config import Settings
    monkeypatch.chdir(tmp_path)  # no .env here: do not read the real keys
    for k in list(__import__("os").environ):
        if k.upper().startswith("GROQ_API_KEY_"):
            monkeypatch.delenv(k)
    s = Settings(groq_api_keys="k1, k2,k3,k1", groq_api_key="k2", _env_file=None)
    assert s.all_groq_keys == ["k1", "k2", "k3"]
