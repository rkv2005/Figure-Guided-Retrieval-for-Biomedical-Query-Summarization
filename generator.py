# ============================================================
# GENERATOR — Groq LLaMA 3.1 8B with N-key rotation
# Fixed across all 7 ablation variants with explicit seed parameter
# ============================================================

import time
import groq as groq_sdk
from groq import Groq, RateLimitError, AuthenticationError, APITimeoutError, APIStatusError


class AblationGenerator:
    """
    Groq LLaMA 3.1 8B generator with N-key rotation.
    Single instance shared across all 7 ablation variants.
    """

    MODEL_ID  = "llama-3.1-8b-instant"
    RPM_LIMIT = 28

    def __init__(self, api_keys: list):
        self.keys        = [k.strip() for k in api_keys if k and k.strip()]
        self.clients     = [Groq(api_key=k) for k in self.keys]
        self.current_idx = 0

        self.request_times = {i: [] for i in range(len(self.keys))}
        self.dead          = {i: False for i in range(len(self.keys))}
        self.rate_limited  = {i: False for i in range(len(self.keys))}
        self.rl_since      = {i: None  for i in range(len(self.keys))}

        if not self.keys:
            raise ValueError("No valid API keys provided.")

        print(f"✅ AblationGenerator ready — {len(self.keys)} keys | {self.MODEL_ID}")


    @property
    def _client(self) -> Groq:
        return self.clients[self.current_idx]

    def _refresh_rate_limits(self):
        now = time.time()
        for i in range(len(self.keys)):
            if self.rate_limited[i] and self.rl_since[i]:
                if now - self.rl_since[i] >= 65:
                    self.rate_limited[i] = False
                    self.rl_since[i]     = None
                    print(f"   ✅ Key [{i+1}] recovered")

    def _available(self) -> list:
        self._refresh_rate_limits()
        return [
            i for i in range(len(self.keys))
            if not self.dead[i] and not self.rate_limited[i]
        ]

    def _rotate(self, reason: str = "") -> bool:
        available = self._available()
        if not available:
            return False
        for step in range(1, len(self.keys) + 1):
            candidate = (self.current_idx + step) % len(self.keys)
            if candidate in available:
                self.current_idx = candidate
                print(f"   🔄 Rotated → key [{candidate+1}] {reason}")
                return True
        return False

    def _wait_for_recovery(self):
        now   = time.time()
        waits = [
            max(0, 65 - (now - self.rl_since[i]))
            for i in range(len(self.keys))
            if self.rate_limited[i] and not self.dead[i] and self.rl_since[i]
        ]
        wait = (min(waits) + 2) if waits else 65
        print(f"   ⏳ All keys rate-limited. Waiting {wait:.1f}s...")
        time.sleep(wait)
        self._refresh_rate_limits()
        if not self._rotate():
            raise RuntimeError("All keys still unavailable after waiting. Aborting.")

    def _throttle(self):
        """Proactive RPM guard — rotate before hitting 429."""
        idx = self.current_idx
        now = time.time()
        self.request_times[idx] = [
            t for t in self.request_times[idx] if now - t < 60
        ]
        if len(self.request_times[idx]) >= self.RPM_LIMIT:
            print(f"   ⏳ RPM ceiling on key [{idx+1}]. Rotating...")
            self.rate_limited[idx] = True
            self.rl_since[idx]     = time.time()
            if not self._rotate(reason="(RPM ceiling)"):
                self._wait_for_recovery()

    def _mark_request(self):
        self.request_times[self.current_idx].append(time.time())


    def generate(self, prompt: str, max_tokens: int = 400, seed: int = None) -> tuple[str, float]:
        """
        Returns (answer_text, elapsed_seconds).
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(5):
            if self.dead[self.current_idx]:
                if not self._rotate(reason="(dead key)"):
                    raise RuntimeError("All API keys permanently invalid.")

            self._throttle()

            try:
                start = time.time()
                
                # Assemble Groq API parameters payload
                params = {
                    "model"       : self.MODEL_ID,
                    "messages"    : messages,
                    "temperature" : 0.7,
                    "max_tokens"  : max_tokens,
                    "top_p"       : 0.9,
                }
                
                # Natively inject seed parameter into Groq's SDK call dictionary
                if seed is not None:
                    params["seed"] = int(seed)

                response = self._client.chat.completions.create(**params)
                
                elapsed = time.time() - start
                self._mark_request()

                self.rate_limited[self.current_idx] = False
                self.rl_since[self.current_idx]     = None

                text = response.choices[0].message.content.strip()
                return text, elapsed

            except RateLimitError:
                print(f"   ⚠️  Key [{self.current_idx+1}] → 429 RateLimitError")
                self.rate_limited[self.current_idx] = True
                self.rl_since[self.current_idx]     = time.time()
                if not self._rotate(reason="(429)"):
                    self._wait_for_recovery()

            except AuthenticationError:
                print(f"   ❌ Key [{self.current_idx+1}] → 401 invalid — marking dead")
                self.dead[self.current_idx] = True
                if not self._rotate(reason="(401 dead)"):
                    raise RuntimeError("All API keys permanently invalid.")

            except APITimeoutError:
                wait = 5 * (2 ** min(attempt, 3))
                print(f"   ⏳ Timeout on attempt {attempt+1}. Waiting {wait}s...")
                self._rotate(reason="(timeout)")
                time.sleep(wait)

            except APIStatusError as e:
                raise RuntimeError(f"Groq API error {e.status_code}: {e.message}")

        raise RuntimeError("Generator failed after 5 attempts.")


    def status(self):
        print(f"\n🔑 Key Status:")
        for i, k in enumerate(self.keys):
            now      = time.time()
            rpm_used = len([t for t in self.request_times[i] if now - t < 60])
            state    = (
                "💀 DEAD"          if self.dead[i]         else
                "🔴 RATE-LIMITED"  if self.rate_limited[i] else
                "✅ OK"
            )
            print(f"   Key [{i+1}] {k[:8]}...  {state}  RPM used: {rpm_used}/{self.RPM_LIMIT}")
