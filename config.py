import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Basic Configuration
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    PRODUCTION = os.getenv('PRODUCTION', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 3000))
    HOST = os.getenv('HOST', '0.0.0.0')

    # Security
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    RATE_LIMIT = int(os.getenv('RATE_LIMIT', 100))  # requests per hour
    MAX_CONCURRENT_USERS = int(os.getenv('MAX_CONCURRENT_USERS', 10))

    # Performance
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))  # 1 hour
    COMPRESSION_QUALITY = int(os.getenv('COMPRESSION_QUALITY', 85))
    MAX_IMAGE_DIMENSION = int(os.getenv('MAX_IMAGE_DIMENSION', 4096))

    # Processing
    DEFAULT_EFFECT = os.getenv('DEFAULT_EFFECT', 'pixelation')
    DEFAULT_STRENGTH = int(os.getenv('DEFAULT_STRENGTH', 7))
    ENABLE_OPENPOSE = os.getenv('ENABLE_OPENPOSE', 'False').lower() == 'true'

    # Paths
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    CACHE_FOLDER = os.getenv('CACHE_FOLDER', 'cache')
    MODELS_FOLDER = os.getenv('MODELS_FOLDER', 'models')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    ENABLE_REQUEST_LOGGING = os.getenv('ENABLE_REQUEST_LOGGING', 'True').lower() == 'true'

    @classmethod
    def init_app(cls, app):
        """Initialize application with configuration"""
        # Ensure required directories exist
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.CACHE_FOLDER, exist_ok=True)
        os.makedirs(cls.MODELS_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)

        # Set Flask configuration
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_CONTENT_LENGTH
        app.config['UPLOAD_FOLDER'] = cls.UPLOAD_FOLDER

        # Configure logging
        import logging
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )

class DevelopmentConfig(Config):
    DEBUG = True
    PRODUCTION = False
    ENABLE_REQUEST_LOGGING = True

class ProductionConfig(Config):
    DEBUG = False
    PRODUCTION = True
    ENABLE_REQUEST_LOGGING = False
    
    # Override with production-specific settings
    @classmethod
    def init_app(cls, app):
        super().init_app(app)
        
        # Additional production setup
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Set up rotating file handler
        file_handler = RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        app.logger.addHandler(file_handler)
        
        # Set up error emails if needed
        if os.getenv('MAIL_SERVER'):
            from logging.handlers import SMTPHandler
            credentials = None
            if os.getenv('MAIL_USERNAME') or os.getenv('MAIL_PASSWORD'):
                credentials = (os.getenv('MAIL_USERNAME'), os.getenv('MAIL_PASSWORD'))
            mail_handler = SMTPHandler(
                mailhost=(os.getenv('MAIL_SERVER'), os.getenv('MAIL_PORT', 25)),
                fromaddr=os.getenv('MAIL_SENDER'),
                toaddrs=[os.getenv('ADMIN_EMAIL')],
                subject='BetaControlAPI Error',
                credentials=credentials,
                secure=() if os.getenv('MAIL_USE_TLS', 'False').lower() == 'true' else None
            )
            mail_handler.setLevel(logging.ERROR)
            app.logger.addHandler(mail_handler)

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Use this to get the active configuration
def get_config():
    return config[os.getenv('FLASK_ENV', 'default')] 