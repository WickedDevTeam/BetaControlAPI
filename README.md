# BetaCensor2

BetaCensor2 is a powerful image censoring API that provides various effects and detection capabilities for sensitive content in images. It offers multiple censoring effects and can detect various body parts and regions that might need censoring.

## Features

- Multiple censoring effects (pixelation, blur, blackbox, ruin, sticker)
- Advanced body part detection
- Face and facial feature detection
- Customizable censoring strength
- RESTful API with Swagger documentation
- Cross-Origin Resource Sharing (CORS) support

## Tech Stack

- Backend:
  - Python
  - Flask
  - OpenCV
  - dlib
  - NudeNet
  - OpenPose (optional)
- Frontend:
  - React
  - TypeScript
  - Vite

## Prerequisites

- Python 3.8+
- Node.js 16+
- OpenCV dependencies
- dlib dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WickedDevTeam/BetaCensor2.git
cd BetaCensor2
```

2. Install backend dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Run the setup script:
```bash
./setup.sh
```

## Usage

1. Start the backend server:
```bash
./start.sh
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Access the application:
- Frontend: http://localhost:5173
- API Documentation: http://localhost:5000/docs

For detailed API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Development

### Project Structure

```
BetaCensor2/
├── backend/
│   ├── app.py
│   └── ...
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── setup.sh
├── start.sh
└── requirements.txt
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team. 