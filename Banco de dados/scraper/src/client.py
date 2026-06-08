"""
Cliente HTTP para a API do Sofascore.

Responsavel por: manter uma sessao com headers de navegador, respeitar
um atraso entre requisicoes (rate limiting educado) e tratar os erros
HTTP mais comuns com novas tentativas.

Anti-ban v3 — correcoes aplicadas:
- BUG CORRIGIDO: 404 agora dorme o delay normal antes de retornar (antes
  retornava imediatamente, causando rajada de requisicoes na Copa Brasil)
- Cooldown automatico a cada COOLDOWN_EVERY requisicoes (pausa longa
  para simular comportamento humano)
- Delay base aumentado: 2.0-4.0s (antes 1.0-2.0s)
- 403: espera 90s * tentativa + rotacao de UA
- 429: espera 90s * tentativa
- Pool de 5 User-Agents com rotacao automatica
- Headers completos sec-ch-ua / sec-fetch-*
"""
import random
import time

import cloudscraper

from config import DELAY_MAX, DELAY_MIN
from .logger import get_logger

log = get_logger("client")

# A cada quantas requisicoes fazer uma pausa longa (simula comportamento humano)
# Sofascore usa janela deslizante estimada em 5-10 min.
# Pausa de 3-5 min a cada 50 reqs garante que a janela expire antes de retomar.
COOLDOWN_EVERY   = 50          # pausa a cada 50 requisicoes (era 80)
COOLDOWN_MIN     = 180.0       # 3 minutos minimos (era 45s)
COOLDOWN_MAX     = 300.0       # 5 minutos maximos (era 90s)

# Pool de User-Agents de browsers reais (Chrome/Edge, Windows/Mac/Linux)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def _build_headers(ua: str) -> dict:
    """Monta headers completos imitando Chrome moderno."""
    return {
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Connection": "keep-alive",
    }


class SofascoreClient:
    """Encapsula a sessao HTTP e a politica de requisicoes."""

    def __init__(self, delay_min: float = DELAY_MIN, delay_max: float = DELAY_MAX):
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.request_count = 0
        self._create_session()

    def _create_session(self) -> None:
        """Cria (ou recria) a sessao cloudscraper com um User-Agent aleatorio."""
        self._current_ua = random.choice(USER_AGENTS)
        self.session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        self.session.headers.update(_build_headers(self._current_ua))
        log.info(f"Sessao criada | UA: {self._current_ua[:60]}...")

    def _rotate_session(self) -> None:
        """Troca para um User-Agent diferente do atual e recria a sessao."""
        outros = [ua for ua in USER_AGENTS if ua != self._current_ua]
        self._current_ua = random.choice(outros)
        self.session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        self.session.headers.update(_build_headers(self._current_ua))
        log.info(f"Sessao rotacionada | UA: {self._current_ua[:60]}...")

    def _polite_sleep(self) -> None:
        """Delay normal entre requisicoes."""
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    def _maybe_cooldown(self) -> None:
        """
        A cada COOLDOWN_EVERY requisicoes faz uma pausa longa.
        Simula o comportamento de um usuario humano que para de olhar
        o site por alguns segundos — reduz drasticamente o risco de ban.
        """
        if self.request_count > 0 and self.request_count % COOLDOWN_EVERY == 0:
            pausa = random.uniform(COOLDOWN_MIN, COOLDOWN_MAX)
            log.info(
                f"[Cooldown] {self.request_count} requisicoes feitas. "
                f"Pausando {pausa:.0f}s para evitar ban..."
            )
            time.sleep(pausa)

    def get_json(self, url: str, max_retries: int = 5):
        """
        Faz uma requisicao GET e devolve o JSON da resposta.

        Tratamento de erros:
        - 200: dorme delay normal, devolve JSON
        - 404: dorme delay curto (CORRIGIDO — antes retornava sem sleep),
               devolve None sem novas tentativas
        - 403: ban de IP — espera 90s*tentativa + troca UA + retry
        - 429: rate limit — espera 90s*tentativa + retry
        - outros: backoff 10s*tentativa ate max_retries
        - excecao de rede: backoff 15s*tentativa ate max_retries
        """
        for attempt in range(1, max_retries + 1):
            try:
                self._maybe_cooldown()
                resp = self.session.get(url, timeout=30)
                self.request_count += 1

                if resp.status_code == 200:
                    self._polite_sleep()
                    return resp.json()

                if resp.status_code == 404:
                    # CORRIGIDO: dorme delay reduzido mesmo no 404
                    # Antes retornava imediatamente, causando rajada de reqs
                    time.sleep(random.uniform(0.5, 1.2))
                    log.warning(f"404 (sem stats): {url.split('/')[-2]}")
                    return None

                if resp.status_code == 403:
                    espera = 90 * attempt  # 90s, 180s, 270s, 360s, 450s
                    log.warning(
                        f"403 (ban) tentativa {attempt}/{max_retries} — "
                        f"aguardando {espera}s e trocando UA..."
                    )
                    time.sleep(espera)
                    self._rotate_session()
                    continue

                if resp.status_code == 429:
                    espera = 90 * attempt
                    log.warning(
                        f"429 (rate limit) tentativa {attempt}/{max_retries} — "
                        f"aguardando {espera}s..."
                    )
                    time.sleep(espera)
                    continue

                espera = 10 * attempt
                log.warning(f"HTTP {resp.status_code} tentativa {attempt} — aguardando {espera}s...")
                time.sleep(espera)

            except Exception as e:
                espera = 15 * attempt
                log.error(f"Erro de rede tentativa {attempt}: {e} — aguardando {espera}s...")
                time.sleep(espera)

        log.error(f"Desisti apos {max_retries} tentativas: {url}")
        return None
