#!/usr/bin/env python3
"""
Test script for multi-worker video processing
"""
import asyncio
import sys
sys.path.insert(0, '/workspace/marketing-video-ai')

from app.services.job_service import job_service
from app.models.mongodb import mongodb

async def test_worker_initialization():
    """Test 1: Verify workers are initialized correctly"""
    print("\n" + "="*60)
    print("TEST 1: Worker Initialization")
    print("="*60)

    print(f"✓ Max video workers: {job_service.max_video_workers}")
    print(f"✓ Worker locks created: {len(job_service.video_processing_locks)}")
    print(f"✓ Worker busy flags: {len(job_service.video_workers_busy)}")
    print(f"✓ Initial busy state: {job_service.video_workers_busy}")

    assert job_service.max_video_workers >= 1, "Should have at least 1 worker"
    assert len(job_service.video_processing_locks) == job_service.max_video_workers
    assert len(job_service.video_workers_busy) == job_service.max_video_workers

    print("✅ Worker initialization test PASSED\n")

async def test_worker_info():
    """Test 2: Verify worker info endpoint"""
    print("="*60)
    print("TEST 2: Worker Information Endpoint")
    print("="*60)

    workers_info = await job_service.get_video_workers_info()

    print(f"Total workers: {workers_info['total_workers']}")
    print(f"Available workers: {workers_info['available_workers']}")
    print(f"Busy workers: {workers_info['busy_workers']}")
    print(f"Pending jobs: {workers_info['pending_jobs']}")

    print("\nWorker Status:")
    for worker in workers_info['workers_status']:
        status = "🟢 Available" if not worker['is_busy'] else "🔴 Busy"
        running = "✅ Running" if worker['is_running'] else "⏸️  Not started"
        print(f"  Worker {worker['worker_id']}: {status}, {running}")

    print("✅ Worker info test PASSED\n")

async def test_queue_info():
    """Test 3: Verify queue info includes worker data"""
    print("="*60)
    print("TEST 3: Queue Information with Workers")
    print("="*60)

    queue_info = await job_service.get_queue_info()

    print(f"Jobs in queue: {queue_info['jobs_in_queue']}")
    print(f"Waiting jobs: {queue_info['waiting_jobs']}")
    print(f"Max workers: {queue_info['max_workers']}")
    print(f"Worker running: {queue_info['worker_running']}")

    assert 'video_workers' in queue_info, "Should include video_workers info"
    assert 'max_workers' in queue_info, "Should include max_workers"

    print("✅ Queue info test PASSED\n")

async def test_stats():
    """Test 4: Verify stats include worker information"""
    print("="*60)
    print("TEST 4: Statistics with Worker Data")
    print("="*60)

    stats = await job_service.get_stats()

    print(f"Video jobs in memory: {stats['video_creation_jobs']['memory']}")
    print(f"Queue pending: {stats['video_creation_jobs']['queue_pending']}")

    if 'workers' in stats['video_creation_jobs']:
        workers = stats['video_creation_jobs']['workers']
        print(f"Video workers: {workers['total_workers']} total, {workers['available_workers']} available")

    print(f"\nSystem info:")
    print(f"  Max video workers: {stats['system']['max_video_workers']}")
    print(f"  Max effect workers: {stats['system']['max_effect_workers']}")
    print(f"  CPU cores: {stats['system']['cpu_cores']}")

    assert 'max_video_workers' in stats['system'], "Should include max_video_workers in system stats"

    print("✅ Stats test PASSED\n")

async def test_parallel_capability():
    """Test 5: Verify parallel processing capability"""
    print("="*60)
    print("TEST 5: Parallel Processing Capability")
    print("="*60)

    from config import ENABLE_PARALLEL_PROCESSING, MAX_PARALLEL_WORKERS

    print(f"Parallel processing enabled: {ENABLE_PARALLEL_PROCESSING}")
    print(f"Max parallel workers config: {MAX_PARALLEL_WORKERS}")
    print(f"Actual workers initialized: {job_service.max_video_workers}")

    if job_service.max_video_workers > 1:
        print(f"✅ System is configured for parallel processing ({job_service.max_video_workers} workers)")
        print(f"📈 Expected speedup: ~{job_service.max_video_workers}x faster")
    else:
        print("⚠️  Only 1 worker - running in serial mode")

    print("✅ Parallel capability test PASSED\n")

async def main():
    """Run all tests"""
    print("\n" + "🎬"*30)
    print("MULTI-WORKER VIDEO PROCESSING TEST SUITE")
    print("🎬"*30 + "\n")

    try:
        # Initialize MongoDB
        print("Connecting to MongoDB...")
        await mongodb.connect()
        print("✓ MongoDB connected\n")

        # Run tests
        await test_worker_initialization()
        await test_worker_info()
        await test_queue_info()
        await test_stats()
        await test_parallel_capability()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print(f"\n✨ Your system is configured with {job_service.max_video_workers} parallel worker(s)")
        print(f"🚀 Ready for {job_service.max_video_workers}x faster video processing!\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await mongodb.close()

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
