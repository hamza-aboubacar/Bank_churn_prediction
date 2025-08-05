# -*- coding: utf-8 -*-
"""
Application Web Flask pour la Prédiction de Désabonnement (Churn) de Clients Bancaires

Cette application Flask permet de :
1. Prédire le désabonnement d'un client individuel via un formulaire.
2. Importer un fichier CSV ou Excel, effectuer des prédictions en masse et afficher/télécharger les résultats.
"""

from flask import Flask, request, render_template, redirect, url_for, send_file, session
import joblib
import pandas as pd
import numpy as np
import os
import io
import logging

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'une_cle_secrete_pour_le_nouveau_projet')

# --- Chargement du modèle ---
try:
    model_path = os.path.join(os.path.dirname(__file__), 'bank_churn_model.pkl')
    model = joblib.load(model_path)
    logging.info("Modèle de prédiction de désabonnement bancaire chargé avec succès.")

    # Obtenir les noms des colonnes d'entrée attendues
    expected_input_columns = model.named_steps['preprocessor'].feature_names_in_
    logging.info(f"Colonnes d'entrée attendues par le modèle: {expected_input_columns.tolist()}")

    # Pour faciliter l'accès aux colonnes numériques et catégorielles originales
    numeric_features_original = []
    categorical_features_original = []
    for transformer_name, transformer, features_list in model.named_steps['preprocessor'].transformers_:
        if transformer_name == 'num':
            numeric_features_original.extend(features_list)
        elif transformer_name == 'cat':
            categorical_features_original.extend(features_list)
    logging.info(f"Colonnes numériques originales: {numeric_features_original}")
    logging.info(f"Colonnes catégorielles originales: {categorical_features_original}")

except FileNotFoundError:
    logging.error("Erreur : Le fichier du modèle 'bank_churn_model.pkl' est introuvable.")
    logging.error("Veuillez d'abord exécuter le script 'model_creation.py' pour générer le modèle.")
    exit()
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}", exc_info=True)
    exit()

# --- Définition des routes de l'application ---

@app.route('/')
def home():
    """
    Affiche la page d'accueil avec le formulaire de saisie des données client individuelles.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Traite les données du formulaire individuel, effectue la prédiction de désabonnement
    et affiche le résultat.
    """
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data = {
                'CreditScore': int(request.form['CreditScore']),
                'Geography': request.form['Geography'],
                'Gender': request.form['Gender'],
                'Age': int(request.form['Age']),
                'Tenure': int(request.form['Tenure']),
                'Balance': float(request.form['Balance']),
                'NumOfProducts': int(request.form['NumOfProducts']),
                'HasCrCard': int(request.form['HasCrCard']),
                'IsActiveMember': int(request.form['IsActiveMember']),
                'EstimatedSalary': float(request.form['EstimatedSalary'])
            }

            input_df = pd.DataFrame([data])
            input_df = input_df[expected_input_columns]

            # Effectuer la prédiction
            prediction_proba = model.predict_proba(input_df)[:, 1]
            prediction = (prediction_proba >= 0.5).astype(int)

            churn_result = "Oui" if prediction[0] == 1 else "Non"
            probability = f"{prediction_proba[0]*100:.2f}%"
            logging.info(f"Prédiction individuelle effectuée: {churn_result} avec probabilité {probability}")

            return render_template('result.html', churn_result=churn_result, probability=probability)

        except ValueError as ve:
            logging.error(f"Erreur de saisie pour prédiction individuelle: {ve}")
            return render_template('result.html', churn_result="Erreur de saisie", probability=f"Veuillez vérifier les valeurs numériques. Erreur: {ve}")
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la prédiction individuelle: {e}", exc_info=True)
            return render_template('result.html', churn_result="Erreur de prédiction", probability=f"Une erreur inattendue est survenue. Erreur: {e}")

@app.route('/upload')
def upload_file_form():
    """
    Affiche le formulaire pour télécharger un fichier dataset.
    """
    return render_template('upload.html')

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    """
    Gère le téléchargement du fichier, effectue des prédictions en masse
    et affiche les résultats.
    """
    if 'file' not in request.files:
        return render_template('upload.html', message='Aucun fichier sélectionné.')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', message='Aucun fichier sélectionné.')
    
    file_extension = file.filename.rsplit('.', 1)[1].lower()

    if file_extension in ['csv', 'xls', 'xlsx']:
        try:
            if file_extension == 'csv':
                df_uploaded = pd.read_csv(io.BytesIO(file.read()))
            elif file_extension in ['xls', 'xlsx']:
                df_uploaded = pd.read_excel(io.BytesIO(file.read()), engine='openpyxl')
            
            logging.info(f"Fichier {file.filename} de type {file_extension} lu avec succès. Lignes: {len(df_uploaded)}")

            df_processed = df_uploaded.copy()

            # Remplir les colonnes manquantes avec NaN et réordonner
            for col in expected_input_columns:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan
            
            df_processed = df_processed[expected_input_columns]

            # Gérer les types de données et les NaN
            for col in expected_input_columns:
                if col in numeric_features_original:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    if df_processed[col].isnull().any():
                        median_val = df_processed[col].median() if not df_processed[col].isnull().all() else 0
                        df_processed[col].fillna(median_val, inplace=True)
                elif col in categorical_features_original:
                    df_processed[col] = df_processed[col].astype(str).replace('nan', 'Unknown')

            predictions_proba = model.predict_proba(df_processed)[:, 1]
            predictions = (predictions_proba >= 0.5).astype(int)
            logging.info(f"Prédictions effectuées pour {len(df_processed)} lignes.")

            df_uploaded['Predicted Churn'] = np.where(predictions == 1, 'Yes', 'No')
            df_uploaded['Churn Probability'] = [f"{p*100:.2f}%" for p in predictions_proba]

            session['predicted_df'] = df_uploaded.to_json(orient='split')

            display_df = df_uploaded.head(10).to_html(classes='table-auto w-full text-left whitespace-no-wrap', index=False)
            
            return render_template('batch_results.html', table=display_df, total_rows=len(df_uploaded))

        except Exception as e:
            logging.error(f"Erreur lors du traitement du fichier {file.filename}: {e}", exc_info=True)
            return render_template('upload.html', message=f"Erreur lors du traitement du fichier : {e}. Veuillez vérifier le format et les données.")
    else:
        logging.warning(f"Format de fichier non supporté: {file.filename}.")
        return render_template('upload.html', message='Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Excel (.xls, .xlsx).')

@app.route('/download_results')
def download_results():
    """
    Permet de télécharger le DataFrame avec les prédictions au format CSV ou Excel.
    """
    if 'predicted_df' in session:
        df_to_download = pd.read_json(session['predicted_df'], orient='split')
        
        output_format = request.args.get('format', 'csv')
        
        if output_format == 'csv':
            csv_buffer = io.StringIO()
            df_to_download.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            return send_file(io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                             mimetype='text/csv',
                             as_attachment=True,
                             download_name='bank_churn_predictions.csv')
        elif output_format == 'xlsx':
            excel_buffer = io.BytesIO()
            df_to_download.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            return send_file(excel_buffer,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             as_attachment=True,
                             download_name='bank_churn_predictions.xlsx')
    
    return redirect(url_for('upload_file_form'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
