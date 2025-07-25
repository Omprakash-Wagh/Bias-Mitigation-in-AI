import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import os
import threading
import webbrowser

# --- Flask web app ---
from flask import Flask, render_template_string, send_file
import io

FLASK_PORT = 5000

class BiasMitigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title('AI Bias Mitigation Desktop App')
        self.df = None
        self.mitigator = None
        self.baseline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.sensitive_train = self.sensitive_test = None
        self.idx_train = self.idx_test = None
        self.web_thread = None
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill='x')

        load_btn = ttk.Button(frame, text='Load CSV', command=self.load_csv)
        load_btn.pack(side='left', padx=5)

        vis_btn = ttk.Button(frame, text='Visualize Bias', command=self.visualize_bias)
        vis_btn.pack(side='left', padx=5)

        mit_btn = ttk.Button(frame, text='Run Mitigation', command=self.run_mitigation)
        mit_btn.pack(side='left', padx=5)

        res_btn = ttk.Button(frame, text='Show Results', command=self.show_results)
        res_btn.pack(side='left', padx=5)

        web_btn = ttk.Button(frame, text='Open Local Website', command=self.launch_website)
        web_btn.pack(side='left', padx=5)

        self.status = tk.StringVar()
        self.status.set('Load a dataset to begin.')
        status_label = ttk.Label(self.root, textvariable=self.status)
        status_label.pack(pady=5)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill='both', expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.status.set(f'Dataset loaded: {os.path.basename(file_path)}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load CSV: {e}')
            self.status.set('Failed to load dataset.')

    def visualize_bias(self):
        if self.df is None:
            messagebox.showwarning('No Data', 'Please load a dataset first.')
            return
        self.clear_plot()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        self.df['Hired'] = self.df['Hired'].astype(int)
        self.df.groupby('Gender')['Hired'].mean().plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Hiring Rate by Gender')
        axes[0].set_ylabel('Hiring Rate')
        self.df.groupby('Ethnicity')['Hired'].mean().plot(kind='bar', ax=axes[1], color='salmon')
        axes[1].set_title('Hiring Rate by Ethnicity')
        axes[1].set_ylabel('Hiring Rate')
        fig.tight_layout()
        self.show_plot(fig)
        self.status.set('Bias visualized.')

    def run_mitigation(self):
        if self.df is None:
            messagebox.showwarning('No Data', 'Please load a dataset first.')
            return
        # Feature engineering
        df = self.df.copy()
        df['Gender_code'] = df['Gender'].astype('category').cat.codes
        df['Ethnicity_code'] = df['Ethnicity'].astype('category').cat.codes
        df['Education_code'] = df['Education_Level'].astype('category').cat.codes
        def skill_count(skills):
            return len(str(skills).split(','))
        df['Skill_Count'] = df['Skills'].apply(skill_count)
        X = df[['Gender_code', 'Ethnicity_code', 'Education_code', 'Years_of_Experience', 'Skill_Count']]
        y = df['Hired']
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.3, random_state=42)
        sensitive_train = df.loc[idx_train, 'Gender']
        sensitive_test = df.loc[idx_test, 'Gender']
        # Baseline model
        baseline = LogisticRegression(max_iter=1000)
        baseline.fit(X_train, y_train)
        # Mitigated model
        mitigator = ExponentiatedGradient(
            LogisticRegression(max_iter=1000),
            constraints=DemographicParity()
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
        # Store for results
        self.baseline = baseline
        self.mitigator = mitigator
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_test = sensitive_test
        self.idx_test = idx_test
        self.status.set('Bias mitigation complete.')

    def show_results(self):
        if self.baseline is None or self.mitigator is None:
            messagebox.showwarning('Not Ready', 'Please run mitigation first.')
            return
        self.clear_plot()
        y_pred_base = self.baseline.predict(self.X_test)
        y_pred_mitigated = self.mitigator.predict(self.X_test)
        # Plot hiring rate by gender before/after
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        pd.DataFrame({'Gender': self.sensitive_test, 'Pred': y_pred_base}).groupby('Gender')['Pred'].mean().plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Baseline Predicted Hiring Rate by Gender')
        pd.DataFrame({'Gender': self.sensitive_test, 'Pred': y_pred_mitigated}).groupby('Gender')['Pred'].mean().plot(kind='bar', ax=axes[1], color='salmon')
        axes[1].set_title('Mitigated Predicted Hiring Rate by Gender')
        fig.tight_layout()
        self.show_plot(fig)
        # Show accuracy and reports
        acc_base = accuracy_score(self.y_test, y_pred_base)
        acc_mitigated = accuracy_score(self.y_test, y_pred_mitigated)
        report_base = classification_report(self.y_test, y_pred_base)
        report_mitigated = classification_report(self.y_test, y_pred_mitigated)
        msg = f'Baseline Accuracy: {acc_base:.2f}\nMitigated Accuracy: {acc_mitigated:.2f}\n\nBaseline Report:\n{report_base}\nMitigated Report:\n{report_mitigated}'
        messagebox.showinfo('Results', msg)
        self.status.set('Results displayed.')

    def launch_website(self):
        if self.df is None or self.baseline is None or self.mitigator is None:
            messagebox.showwarning('Not Ready', 'Please load data and run mitigation first.')
            return
        if self.web_thread and self.web_thread.is_alive():
            webbrowser.open(f'http://localhost:{FLASK_PORT}')
            messagebox.showinfo('Website', f'Website already running at http://localhost:{FLASK_PORT}\n(If it does not open automatically, copy and paste this link into your browser.)')
            return
        self.web_thread = threading.Thread(target=self._run_flask, daemon=True)
        self.web_thread.start()
        webbrowser.open(f'http://localhost:{FLASK_PORT}')
        messagebox.showinfo('Website', f'Website running at http://localhost:{FLASK_PORT}\n(If it does not open automatically, copy and paste this link into your browser.)')

    def _run_flask(self):
        app = Flask(__name__)
        self._prepare_web_data()
        @app.route('/')
        def index():
            return render_template_string(self._web_template(),
                acc_base=self.web_acc_base,
                acc_mitigated=self.web_acc_mitigated,
                report_base=self.web_report_base,
                report_mitigated=self.web_report_mitigated,
                explanation=self._explanation_html()
            )
        @app.route('/plot/<which>.png')
        def plot_png(which):
            fig = self._web_plot(which)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype='image/png')
        app.run(port=FLASK_PORT, debug=False, use_reloader=False)

    def _prepare_web_data(self):
        y_pred_base = self.baseline.predict(self.X_test)
        y_pred_mitigated = self.mitigator.predict(self.X_test)
        self.web_acc_base = accuracy_score(self.y_test, y_pred_base)
        self.web_acc_mitigated = accuracy_score(self.y_test, y_pred_mitigated)
        self.web_report_base = classification_report(self.y_test, y_pred_base)
        self.web_report_mitigated = classification_report(self.y_test, y_pred_mitigated)
        self.web_sensitive_test = self.sensitive_test
        self.web_y_pred_base = y_pred_base
        self.web_y_pred_mitigated = y_pred_mitigated

    def _web_plot(self, which):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if which == 'gender':
            pd.DataFrame({'Gender': self.web_sensitive_test, 'Pred': self.web_y_pred_base}).groupby('Gender')['Pred'].mean().plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title('Baseline Predicted Hiring Rate by Gender')
            pd.DataFrame({'Gender': self.web_sensitive_test, 'Pred': self.web_y_pred_mitigated}).groupby('Gender')['Pred'].mean().plot(kind='bar', ax=axes[1], color='salmon')
            axes[1].set_title('Mitigated Predicted Hiring Rate by Gender')
        elif which == 'ethnicity':
            pd.DataFrame({'Ethnicity': self.df.loc[self.idx_test, 'Ethnicity'], 'Pred': self.web_y_pred_base}).groupby('Ethnicity')['Pred'].mean().plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title('Baseline Predicted Hiring Rate by Ethnicity')
            pd.DataFrame({'Ethnicity': self.df.loc[self.idx_test, 'Ethnicity'], 'Pred': self.web_y_pred_mitigated}).groupby('Ethnicity')['Pred'].mean().plot(kind='bar', ax=axes[1], color='salmon')
            axes[1].set_title('Mitigated Predicted Hiring Rate by Ethnicity')
        fig.tight_layout()
        return fig

    def _web_template(self):
        return '''
        <html>
        <head>
        <title>AI Bias Mitigation Results</title>
        <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Roboto', Arial, sans-serif; background: #f7f9fa; margin: 0; padding: 0; }
            .header { background: #2d6cdf; color: white; padding: 24px 0 16px 0; text-align: center; box-shadow: 0 2px 8px #0001; }
            .container { max-width: 900px; margin: 32px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px #0002; padding: 32px 40px; }
            h1 { margin-top: 0; font-size: 2.5em; letter-spacing: 1px; }
            h2 { color: #2d6cdf; margin-top: 2em; }
            .card { background: #f2f6fc; border-radius: 8px; padding: 18px 24px; margin: 18px 0; box-shadow: 0 2px 8px #0001; }
            .result { font-size: 1.2em; margin: 10px 0; }
            pre { background: #f7f7f7; border-radius: 6px; padding: 12px; overflow-x: auto; }
            img { border-radius: 8px; box-shadow: 0 2px 8px #0001; margin: 12px 0; }
            .footer { text-align: center; color: #888; margin-top: 40px; font-size: 0.95em; }
            a { color: #2d6cdf; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
        </head>
        <body>
        <div class="header">
            <h1>AI Bias Mitigation Results</h1>
        </div>
        <div class="container">
        <h2>How This Works</h2>
        <div class="card">{{ explanation|safe }}</div>
        <h2>Results</h2>
        <div class="card">
            <div class="result"><b>Baseline Model Accuracy:</b> <span style="color:#2d6cdf">{{ acc_base }}</span></div>
            <div class="result"><b>Mitigated Model Accuracy:</b> <span style="color:#27ae60">{{ acc_mitigated }}</span></div>
        </div>
        <div class="card">
            <b>Baseline Classification Report:</b><pre>{{ report_base }}</pre>
            <b>Mitigated Classification Report:</b><pre>{{ report_mitigated }}</pre>
        </div>
        <h2>Predicted Hiring Rate by Gender</h2>
        <div class="card"><img src="/plot/gender.png" width="700"></div>
        <h2>Predicted Hiring Rate by Ethnicity</h2>
        <div class="card"><img src="/plot/ethnicity.png" width="700"></div>
        </div>
        <div class="footer">
            <hr>
            Locally hosted with Flask. Close this window to stop the server.<br>
            <a href="https://fairlearn.org/" target="_blank">Learn more about Fairlearn</a>
        </div>
        </body></html>
        '''

    def _explanation_html(self):
        return '''
        <p>This tool demonstrates how to detect and mitigate bias in AI models using a real-world dataset.<br>
        <ul>
        <li><b>Step 1:</b> Load your dataset (CSV) with columns for Gender, Ethnicity, Education, Experience, Skills, and Hired outcome.</li>
        <li><b>Step 2:</b> Visualize bias in hiring rates by gender and ethnicity.</li>
        <li><b>Step 3:</b> Run bias mitigation using Fairlearn's ExponentiatedGradient with DemographicParity constraint.</li>
        <li><b>Step 4:</b> Compare the baseline and mitigated model results, including accuracy and fairness metrics.</li>
        </ul>
        <b>How it works:</b> The app uses logistic regression as a baseline, then applies a fairness constraint to reduce bias in predictions. Plots and metrics help you understand the impact of mitigation.</p>
        '''

    def clear_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    def show_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

if __name__ == '__main__':
    root = tk.Tk()
    app = BiasMitigationApp(root)
    root.mainloop() 