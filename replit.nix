{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.flask
    pkgs.python3Packages.flask-cors
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
    pkgs.nodejs_20
    pkgs.nodePackages.npm
    pkgs.nodePackages.typescript
    pkgs.nodePackages.typescript-language-server
    pkgs.bashInteractive
    pkgs.cmake
    pkgs.gcc
    pkgs.gnumake
    pkgs.git
  ];
  env = {
    PYTHONBIN = "${pkgs.python3}/bin/python3";
    LANG = "en_US.UTF-8";
  };
} 