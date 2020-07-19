import uvicorn
import subprocess, os

## run ./run_tfserving in another terminal

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
