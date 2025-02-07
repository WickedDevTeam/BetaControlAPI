{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.flask
    pkgs.python3Packages.opencv4
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pillow
    pkgs.python3Packages.dlib
    pkgs.python3Packages.requests
    pkgs.python3Packages.flask-cors
    pkgs.python3Packages.psutil
    pkgs.python3Packages.python-dotenv
    pkgs.python3Packages.gunicorn
    pkgs.nodejs_20
    pkgs.nodePackages.npm
    pkgs.bashInteractive
    pkgs.cmake
    pkgs.gcc
    pkgs.gnumake
    pkgs.git
  ];
} 