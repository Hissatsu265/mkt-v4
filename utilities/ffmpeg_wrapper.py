"""
FFmpeg wrapper with proper signal handling and error recovery.
This module provides robust FFmpeg execution to prevent "hard exiting" errors.
"""
import subprocess
import signal
import logging
import time
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class FFmpegWrapper:
    """Wrapper for FFmpeg subprocess calls with proper signal handling."""

    @staticmethod
    def run_ffmpeg(
        cmd: List[str],
        timeout: Optional[int] = None,
        capture_output: bool = True,
        retry_count: int = 2,
        log_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run FFmpeg command with proper signal handling and retry logic.

        Args:
            cmd: FFmpeg command as list of strings
            timeout: Optional timeout in seconds
            capture_output: Whether to capture stdout/stderr
            retry_count: Number of retries on failure
            log_output: Whether to log FFmpeg output

        Returns:
            CompletedProcess object

        Raises:
            subprocess.CalledProcessError: If FFmpeg fails after all retries
        """
        attempt = 0
        last_error = None

        while attempt <= retry_count:
            try:
                if attempt > 0:
                    logger.info(f"Retrying FFmpeg command (attempt {attempt + 1}/{retry_count + 1})...")
                    time.sleep(1)  # Brief delay before retry

                # Use Popen for better signal control
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE if capture_output else None,
                    stderr=subprocess.PIPE if capture_output else None,
                    text=True,
                    preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in child
                )

                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    returncode = process.returncode

                    if log_output and stderr and returncode != 0:
                        logger.error(f"FFmpeg stderr: {stderr[:500]}")

                    if returncode == 0:
                        return subprocess.CompletedProcess(
                            args=cmd,
                            returncode=returncode,
                            stdout=stdout,
                            stderr=stderr
                        )
                    else:
                        last_error = subprocess.CalledProcessError(
                            returncode=returncode,
                            cmd=cmd,
                            output=stdout,
                            stderr=stderr
                        )

                except subprocess.TimeoutExpired:
                    logger.warning(f"FFmpeg command timed out after {timeout}s")
                    process.kill()
                    process.wait()
                    last_error = subprocess.TimeoutExpired(cmd, timeout)

            except Exception as e:
                logger.error(f"FFmpeg execution error: {str(e)}")
                last_error = e

            attempt += 1

        # All retries failed
        if last_error:
            if isinstance(last_error, subprocess.CalledProcessError):
                raise last_error
            else:
                raise subprocess.CalledProcessError(
                    returncode=-1,
                    cmd=cmd,
                    output=str(last_error)
                )

        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="Unknown error")

    @staticmethod
    def run_ffmpeg_safe(
        cmd: List[str],
        timeout: Optional[int] = None,
        log_output: bool = True
    ) -> Dict[str, any]:
        """
        Run FFmpeg command and return result dict (doesn't raise on error).

        Args:
            cmd: FFmpeg command as list of strings
            timeout: Optional timeout in seconds
            log_output: Whether to log FFmpeg output

        Returns:
            Dict with keys: success (bool), returncode (int), stdout (str), stderr (str), error (str)
        """
        try:
            result = FFmpegWrapper.run_ffmpeg(
                cmd=cmd,
                timeout=timeout,
                capture_output=True,
                retry_count=1,
                log_output=log_output
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "error": None
            }
        except Exception as e:
            logger.error(f"FFmpeg safe execution failed: {str(e)}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "error": str(e)
            }


# Convenience function
def run_ffmpeg_command(cmd: List[str], timeout: Optional[int] = None, **kwargs) -> subprocess.CompletedProcess:
    """
    Convenience function to run FFmpeg with proper signal handling.

    Args:
        cmd: FFmpeg command as list of strings
        timeout: Optional timeout in seconds
        **kwargs: Additional arguments passed to FFmpegWrapper.run_ffmpeg

    Returns:
        CompletedProcess object
    """
    return FFmpegWrapper.run_ffmpeg(cmd, timeout=timeout, **kwargs)
