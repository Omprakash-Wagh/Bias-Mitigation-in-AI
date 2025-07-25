# AI Bias Mitigation

This project demonstrates how to detect and mitigate bias in AI models using Python. It features a desktop app (Tkinter) that allows you to:
- Load your own dataset (CSV)
- Visualize bias in hiring rates by gender and ethnicity
- Apply bias mitigation using Fairlearn
- View results and fairness metrics
- Optionally, launch a local website (Flask) to view results and explanations in your browser

## Project Structure
- `desktop_app.py`: Main desktop application (Tkinter GUI + Flask website)
- `biased_ai_dataset.csv`: Example dataset
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## How to Use
1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   (If using a virtual environment, activate it first.)

2. **Run the desktop app**
   ```
   python desktop_app.py
   ```
   or (if using a venv):
   ```
   .venv\Scripts\python.exe desktop_app.py
   ```

3. **In the app:**
   - Click "Load CSV" to load your dataset (or use the provided example)
   - Click "Visualize Bias" to see bias in the data
   - Click "Run Mitigation" to apply bias mitigation
   - Click "Show Results" to view accuracy and fairness metrics
   - Click "Open Local Website" to view results and explanations in your browser (http://localhost:5000)

## Dependencies
See `requirements.txt` for required packages (Tkinter is included with Python).

## Notes
- The app uses logistic regression as a baseline, then applies Fairlearn's ExponentiatedGradient with DemographicParity to mitigate bias.
- All results and plots are available both in the desktop app and the local website.

---
Made with ❤️ using Python, Tkinter, Flask, scikit-learn, matplotlib, and Fairlearn. 