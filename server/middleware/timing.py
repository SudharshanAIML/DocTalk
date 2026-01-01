from fastapi import Request

async def timing_middleware(request: Request, call_next):
    import time

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    response.headers["X-Process-Time"] = str(round(duration, 4))
    return response
