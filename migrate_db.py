import time
from app import app, db
import os
import shutil

def migrate_database():
    with app.app_context():
        # Backup old database if exists
        if os.path.exists('instance/study.db'):
            try:
                # Try normal rename first
                os.rename('instance/study.db', 'instance/study.db.backup')
            except PermissionError:
                # If locked, use copy + delete
                shutil.copy2('instance/study.db', 'instance/study.db.backup')
                time.sleep(1)  # Wait a moment
                try:
                    os.remove('instance/study.db')
                except:
                    pass  # Continue even if delete fails
        
        # Create fresh database with current schema
        db.create_all()
        print("Database migrated successfully")

if __name__ == '__main__':
    migrate_database()
