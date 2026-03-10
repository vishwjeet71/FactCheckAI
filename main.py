import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from prototype import Prototype

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Pipeline is I/O-heavy (scraping + LLM calls). 150 s is generous but not infinite.
# If it hasn't returned by then, tell the user to retry rather than hanging forever.
PIPELINE_TIMEOUT_SECONDS = 150

# ThreadPoolExecutor lets us run the blocking Prototype.run() without freezing
# FastAPI's async event loop. max_workers=4 handles 4 simultaneous requests.
THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FactCheck AI",
    description="AI-powered news fact-checking and trust analysis",
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Request / Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    text:           str
    inputType:      str   # "query" | "article_link"
    searchApiKey:   str   # user's search API key
    searchProvider: str   # "serper.dev" | "gnews" | "publicapi.dev"
    llmApiKey:      str   # user's LLM API key
    llmProvider:    str   # "anthropic" | "openai" | "google" | "groq"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 5000:
            raise ValueError("text exceeds 5 000 characters")
        return v

    @field_validator("inputType")
    @classmethod
    def valid_input_type(cls, v: str) -> str:
        allowed = {"query", "article_link"}
        if v not in allowed:
            raise ValueError(f"inputType must be one of {allowed}")
        return v

    @field_validator("llmProvider")
    @classmethod
    def valid_llm_provider(cls, v: str) -> str:
        allowed = {"anthropic", "openai", "google", "groq"}
        if v not in allowed:
            raise ValueError(f"llmProvider must be one of {allowed}")
        return v


class ChatResponse(BaseModel):
    response:  str
    inputType: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    """
    Runs the Prototype pipeline end-to-end and streams the LLM explanation
    back to the frontend as a plain string inside ChatResponse.

    Flow:
      1. Build Prototype with the user-supplied search API key.
         key_provider is always "superdev" — that is the internal name used by
         Apicaller regardless of which search provider the user picked in the UI.
      2. Run the blocking pipeline in a thread-pool so FastAPI stays responsive.
      3. Enforce a 150-second timeout. If it fires, return a user-friendly
         "please try again" message instead of hanging the request forever.
      4. Catch any unexpected exception and return a readable error message.
         The pipeline itself already converts most internal errors into LLM
         explanations, so this outer catch is only a safety net.
    """
    logger.info(
        "Chat request | type=%s | llm=%s | search=%s | text=%.120s",
        payload.inputType,
        payload.llmProvider,
        payload.searchProvider,
        payload.text,
    )

    # Prototype's internal search provider name is always "superdev".
    # The UI label ("serper.dev", "gnews", etc.) is only for display;
    # the actual API call routing is done inside Apicaller.
    KEY_PROVIDER = "superdev"

    pipeline = Prototype(payload.searchApiKey, KEY_PROVIDER)

    def _run_pipeline() -> str:
        """Blocking call — executed in the thread pool."""
        return pipeline.run(
            payload.text,
            payload.inputType,
            llm_api_key=payload.llmApiKey,
            llm_provider=payload.llmProvider,
        )

    loop = asyncio.get_event_loop()

    try:
        result: str = await asyncio.wait_for(
            loop.run_in_executor(THREAD_POOL, _run_pipeline),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
        logger.info("Pipeline finished successfully | type=%s", payload.inputType)
        return ChatResponse(response=result, inputType=payload.inputType)

    except asyncio.TimeoutError:
        logger.warning(
            "Pipeline timed out after %ds | type=%s | text=%.80s",
            PIPELINE_TIMEOUT_SECONDS,
            payload.inputType,
            payload.text,
        )
        return ChatResponse(
            response=(
                "⏱️ Analysis timed out.\n\n"
                "The pipeline ran for over 2.5 minutes without finishing. "
                "This usually happens when sources are slow to load or the claim is very broad.\n\n"
                "Please try again with a shorter or more specific query."
            ),
            inputType=payload.inputType,
        )

    except Exception as exc:
        # This fires only if something completely unexpected blows up —
        # the pipeline's own error handling converts most failures into
        # user-readable LLM explanations before reaching this point.
        logger.exception("Unhandled exception in pipeline: %s", exc)
        return ChatResponse(
            response=(
                "⚠️ Something unexpected went wrong.\n\n"
                "The analysis could not complete. Please check that your API keys "
                "are correct and try again.\n\n"
                f"Technical detail: {str(exc)[:200]}"
            ),
            inputType=payload.inputType,
        )


# ── Health / Info ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "FactCheck AI", "version": "1.1.0"}


@app.get("/about")
async def about():
    return {
        "name":        "FactCheck AI",
        "description": "AI-powered news verification and trust analysis",
        "version":     "1.1.0",
        "modes":       ["query verification", "article URL analysis"],
    }