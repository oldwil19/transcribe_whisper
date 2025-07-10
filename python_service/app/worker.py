import os
import logging
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.worker import Worker

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Redis
redis_conn = Redis.from_url(os.getenv("REDIS_URL"))
task_queue = Queue(connection=redis_conn)

def start_worker():
    """Inicia un worker RQ para procesar tareas en segundo plano"""
    worker = Worker([task_queue], connection=redis_conn)
    worker.work()

def enqueue_task(task_func, *args, **kwargs):
    """Encola una tarea para ser procesada en segundo plano"""
    job = task_queue.enqueue(task_func, *args, **kwargs)
    logger.info(f"Tarea encolada con ID: {job.id}")
    return job.id

def get_job_status(job_id):
    """Obtiene el estado de una tarea"""
    job = Job.fetch(job_id, connection=redis_conn)
    return {
        "id": job.id,
        "status": job.get_status(),
        "result": job.result,
        "error": job.exc_info
    }
