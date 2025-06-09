from django.apps import AppConfig
from .views import update_crowd_data
import os
import time, threading


class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        if os.environ.get('RUN_MAIN', None) != 'true':
            return
        
        def run_continuous_updates():
            while True:
                print("=== Updating Crowd Data ===")
                update_crowd_data()
                time.sleep(60)

        thread = threading.Thread(target=run_continuous_updates, daemon=True)
        thread.start()
