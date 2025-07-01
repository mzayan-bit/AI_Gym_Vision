# AI Gym Vision 🏋️

An AI-powered fitness coach that uses computer vision to analyze gym exercises in real-time. This project was developed by interns at Integration Xperts.

## Features
- **Machine Exercise Analysis:** Provides real-time form correction, rep counting, and scoring for various gym machine exercises.
- **Non-Machine Exercise Analysis:** Tracks push-ups and squats, providing feedback and counting reps.
- **Machine Efficiency Analysis:** Determines if gym equipment is occupied and being used efficiently.
- **Detailed Reporting:** Generates downloadable session reports in Excel and visual graphs of performance over time.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mzayan-bit/app_development.git](https://github.com/mzayan-bit/app_development.git)
    cd AI-Gym-Vision
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Files:**
    The required YOLO model files (`gym.pt`, `yolov8n.pt`) are not included in this repository due to their size. Please download them from [Provide a Link here, e.g., Google Drive, or just a note to find them] and place them in the `trained_model/` directory.

## How to Run

Run the Streamlit application from your terminal:
```bash
streamlit run ai_gym.py