import os
import time
import logging
from typing import Dict, Any, List
from config.settings import CACHE_DIR, BASE_DIR

logger = logging.getLogger(__name__)

CACHE_MARKET_DIR = os.path.join(BASE_DIR, "data", "cache_market")

# Configuración por defecto de retención de caché
DEFAULT_MAX_FILES = 3          # Mantener como máximo los 3 últimos hechos de empresas analizadas
DEFAULT_MAX_AGE_DAYS = 2       # Eliminar archivos con más de 2 días de antigüedad
DEFAULT_MAX_SIZE_MB = 20.0     # Límite máximo de tamaño total de caché en MB


class CacheManager:
    """
    Gestiona la limpieza periódica y automática del almacenamiento en caché
    de datos de la SEC y de mercado para evitar acumulación excesiva de espacio en disco.
    """

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Calcula el número de archivos y el tamaño total ocupado por las carpetas de caché."""
        dirs = [CACHE_DIR, CACHE_MARKET_DIR]
        total_files = 0
        total_bytes = 0
        file_details = []

        for d in dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                fpath = os.path.join(d, f)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    mtime = os.path.getmtime(fpath)
                    total_files += 1
                    total_bytes += size
                    file_details.append({
                        "path": fpath,
                        "name": f,
                        "dir": os.path.basename(d),
                        "size_mb": size / (1024 * 1024),
                        "mtime": mtime
                    })

        return {
            "total_files": total_files,
            "total_size_mb": total_bytes / (1024 * 1024),
            "files": file_details
        }

    @staticmethod
    def auto_clean(
        max_files: int = DEFAULT_MAX_FILES,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        max_size_mb: float = DEFAULT_MAX_SIZE_MB,
        protected_files: List[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta una limpieza inteligente del caché:
        1. Elimina archivos de facts más antiguos que `max_age_days`.
        2. Conserva como máximo `max_files` empresas en facts (LRU: elimina los menos recientes).
        3. Si el tamaño total sigue superando `max_size_mb`, elimina los más antiguos hasta cumplir el límite.
        4. Protege archivos clave del sistema (como company_tickers.json).
        """
        if protected_files is None:
            protected_files = ["company_tickers.json"]

        now = time.time()
        max_age_seconds = max_age_days * 86400
        deleted_files = []
        freed_bytes = 0

        dirs_to_clean = [
            (CACHE_DIR, ["_facts.json"]),
            (CACHE_MARKET_DIR, ["_market.json"])
        ]

        for directory, patterns in dirs_to_clean:
            if not os.path.exists(directory):
                continue

            # Listar archivos objetivo (excluyendo protegidos)
            target_files = []
            for fname in os.listdir(directory):
                if fname in protected_files:
                    continue
                if any(fname.endswith(p) for p in patterns) or patterns == ["*"]:
                    fpath = os.path.join(directory, fname)
                    if os.path.isfile(fpath):
                        target_files.append({
                            "path": fpath,
                            "name": fname,
                            "size": os.path.getsize(fpath),
                            "mtime": os.path.getmtime(fpath)
                        })

            # Ordenar por fecha de modificación (más reciente primero)
            target_files.sort(key=lambda x: x["mtime"], reverse=True)

            # 1. Regla por antigüedad (> max_age_days)
            for item in list(target_files):
                if (now - item["mtime"]) > max_age_seconds:
                    try:
                        os.remove(item["path"])
                        deleted_files.append(item["name"])
                        freed_bytes += item["size"]
                        target_files.remove(item)
                    except OSError as e:
                        logger.warning(f"[CacheManager] No se pudo eliminar {item['name']}: {e}")

            # 2. Regla por número máximo de archivos (LRU)
            if len(target_files) > max_files:
                files_to_remove = target_files[max_files:]
                for item in files_to_remove:
                    try:
                        os.remove(item["path"])
                        deleted_files.append(item["name"])
                        freed_bytes += item["size"]
                        target_files.remove(item)
                    except OSError as e:
                        logger.warning(f"[CacheManager] No se pudo eliminar {item['name']}: {e}")

        # 3. Regla por tamaño total máximo
        stats = CacheManager.get_stats()
        if stats["total_size_mb"] > max_size_mb:
            # Ordenar todos los archivos no protegidos por mtime ascendente (más viejos primero)
            candidates = [f for f in stats["files"] if f["name"] not in protected_files]
            candidates.sort(key=lambda x: x["mtime"])

            for item in candidates:
                if stats["total_size_mb"] <= max_size_mb:
                    break
                try:
                    os.remove(item["path"])
                    deleted_files.append(item["name"])
                    freed_bytes += int(item["size_mb"] * 1024 * 1024)
                    stats["total_size_mb"] -= item["size_mb"]
                except OSError as e:
                    logger.warning(f"[CacheManager] No se pudo eliminar {item['name']}: {e}")

        freed_mb = freed_bytes / (1024 * 1024)
        if deleted_files:
            logger.info(f"[CacheManager] Limpieza automática completada: {len(deleted_files)} archivos eliminados ({freed_mb:.2f} MB liberados).")
        else:
            logger.debug("[CacheManager] El caché se encuentra dentro de los límites óptimos.")

        return {
            "deleted_files": deleted_files,
            "freed_mb": freed_mb,
            "current_stats": CacheManager.get_stats()
        }

    @staticmethod
    def purge_all(keep_mapping: bool = True) -> Dict[str, Any]:
        """
        Elimina inmediatamente todos los archivos cacheados de empresas.
        Si keep_mapping es True, mantiene el archivo de mapeo CIK (company_tickers.json).
        """
        protected = ["company_tickers.json"] if keep_mapping else []
        dirs = [CACHE_DIR, CACHE_MARKET_DIR]
        deleted = []
        freed_bytes = 0

        for d in dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if f in protected:
                    continue
                fpath = os.path.join(d, f)
                if os.path.isfile(fpath):
                    try:
                        size = os.path.getsize(fpath)
                        os.remove(fpath)
                        deleted.append(f)
                        freed_bytes += size
                    except OSError as e:
                        logger.warning(f"[CacheManager] Error purgando {f}: {e}")

        freed_mb = freed_bytes / (1024 * 1024)
        logger.info(f"[CacheManager] Purga total de caché completada: {len(deleted)} archivos ({freed_mb:.2f} MB liberados).")
        return {
            "deleted_files": deleted,
            "freed_mb": freed_mb,
            "current_stats": CacheManager.get_stats()
        }
