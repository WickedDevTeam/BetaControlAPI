{ pkgs }: {
  deps = [
    # Python and core packages
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.wheel

    # Web framework and related
    pkgs.python3Packages.flask
    pkgs.python3Packages.flask-cors
    pkgs.python3Packages.flask-restx
    pkgs.python3Packages.gunicorn
    pkgs.python3Packages.python-dotenv
    pkgs.python3Packages.requests

    # Image processing
    pkgs.python3Packages.opencv4
    pkgs.python3Packages.pillow
    pkgs.python3Packages.numpy
    pkgs.python3Packages.dlib
    pkgs.python3Packages.tensorflow
    pkgs.python3Packages.keras
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.scikit-image
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.pandas

    # Utilities
    pkgs.python3Packages.psutil
    pkgs.python3Packages.tqdm
    pkgs.python3Packages.colorlog
    pkgs.python3Packages.pyyaml

    # Node.js and frontend tools
    pkgs.nodejs_20
    pkgs.nodePackages.npm
    pkgs.nodePackages.typescript
    pkgs.nodePackages.typescript-language-server

    # System tools
    pkgs.bashInteractive
    pkgs.cmake
    pkgs.gcc
    pkgs.gnumake
    pkgs.git
    pkgs.bzip2
    pkgs.curl
  ];

  env = {
    PYTHONBIN = "${pkgs.python3}/bin/python3";
    LANG = "en_US.UTF-8";
    LD_LIBRARY_PATH = "${pkgs.opencv4}/lib:${pkgs.dlib}/lib:$LD_LIBRARY_PATH";
    PYTHONPATH = "${pkgs.python3Packages.opencv4}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.dlib}/${pkgs.python3.sitePackages}:$PYTHONPATH";
    OPENCV_PYTHON_SITE_PACKAGES = "${pkgs.python3Packages.opencv4}/${pkgs.python3.sitePackages}";
  };
} 