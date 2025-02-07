{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.flask
    pkgs.python3Packages.flask-cors
    pkgs.python3Packages.flask-restx
    pkgs.python3Packages.opencv4
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pillow
    pkgs.python3Packages.dlib
    pkgs.python3Packages.requests
    pkgs.python3Packages.psutil
    pkgs.python3Packages.python-dotenv
    pkgs.python3Packages.gunicorn
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.wheel
    pkgs.python3Packages.tensorflow
    pkgs.python3Packages.keras
    pkgs.python3Packages.colorlog
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.scipy
    pkgs.python3Packages.pandas
    pkgs.nodejs_20
    pkgs.nodePackages.npm
    pkgs.nodePackages.typescript
    pkgs.nodePackages.typescript-language-server
    pkgs.bashInteractive
    pkgs.cmake
    pkgs.gcc
    pkgs.gnumake
    pkgs.git
    pkgs.bzip2
    pkgs.libGL
    pkgs.glib
    pkgs.xorg.libX11
  ];
  env = {
    PYTHONBIN = "${pkgs.python3}/bin/python3";
    LANG = "en_US.UTF-8";
    LD_LIBRARY_PATH = "${pkgs.opencv4}/lib:${pkgs.dlib}/lib:${pkgs.libGL}/lib:$LD_LIBRARY_PATH";
    PYTHONPATH = "${pkgs.python3Packages.opencv4}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.dlib}/${pkgs.python3.sitePackages}:$PYTHONPATH";
    OPENCV_IO_MAX_IMAGE_PIXELS = "pow(2,40)";
  };
} 