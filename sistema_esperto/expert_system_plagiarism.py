"""
Sistema Esperto per Rilevamento Plagio - Interfaccia Grafica
Integra Knowledge Base Prolog con interfaccia Python/Tkinter

Autore: [Raffaella Nitti]
Matricola: [796132]
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pyswip import Prolog
import os

class PlagiarismExpertSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Plagiarism Detection Expert System")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Inizializza Prolog
        self.prolog = Prolog()
        self.load_knowledge_base()
        
        # Crea interfaccia
        self.create_widgets()
        
    def load_knowledge_base(self):
        """Carica la Knowledge Base Prolog"""
        kb_path = os.path.join(os.path.dirname(__file__), 'plagiarism_rules.pl')
        try:
            self.prolog.consult(kb_path)
            print("✅ Knowledge Base caricata con successo!")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare KB:\n{str(e)}")
            
    def create_widgets(self):
        """Crea l'interfaccia grafica"""
        
        # Titolo
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x')
        
        title_label = tk.Label(
            title_frame,
            text="🔍 SISTEMA ESPERTO - RILEVAMENTO PLAGIO",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Frame principale
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # ===== SEZIONE INPUT =====
        input_frame = tk.LabelFrame(
            main_frame,
            text="📝 Parametri di Input",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        input_frame.pack(fill='x', pady=(0, 15))
        
        # Similarità
        self.create_slider(input_frame, "Similarità (0-1):", 0, 1, 0.01, 0.5, 0)
        
        # Percentuale copiata
        self.create_slider(input_frame, "Percentuale copiata (%):", 0, 100, 1, 20, 1)
        
        # Similarità concettuale
        self.create_slider(input_frame, "Similarità concettuale (0-1):", 0, 1, 0.01, 0.4, 2)
        
        # Dropdown choices
        row = 3
        
        # Citazione presente
        self.create_dropdown(input_frame, "Citazione presente:", 
                           ['no', 'yes', 'corretta'], 'no', row)
        row += 1
        
        # Match esatto
        self.create_dropdown(input_frame, "Match esatto:", 
                           ['yes', 'no'], 'no', row)
        row += 1
        
        # Fonti multiple
        self.create_dropdown(input_frame, "Fonti multiple:", 
                           ['yes', 'no'], 'no', row)
        row += 1
        
        # Qualità parafrasi
        self.create_dropdown(input_frame, "Qualità parafrasi:", 
                           ['scarsa', 'media', 'buona'], 'media', row)
        row += 1
        
        # Fonte dichiarata
        self.create_dropdown(input_frame, "Fonte dichiarata:", 
                           ['yes', 'no'], 'yes', row)
        row += 1
        
        # Idea originale
        self.create_dropdown(input_frame, "Idea originale:", 
                           ['yes', 'no'], 'yes', row)
        row += 1
        
        # Disciplina
        self.create_dropdown(input_frame, "Disciplina:", 
                           ['informatica', 'medicina', 'filosofia', 
                            'ingegneria', 'letteratura'], 'informatica', row)
        row += 1
        
        # Livello studio
        self.create_dropdown(input_frame, "Livello studio:", 
                           ['bachelor', 'master', 'phd'], 'bachelor', row)
        
        # ===== BOTTONE ANALISI =====
        analyze_btn = tk.Button(
            main_frame,
            text="🔍 ANALIZZA",
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.analyze,
            cursor='hand2',
            height=2
        )
        analyze_btn.pack(fill='x', pady=(0, 15))
        
        # ===== SEZIONE OUTPUT =====
        output_frame = tk.LabelFrame(
            main_frame,
            text="📊 Risultati Analisi",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        output_frame.pack(fill='both', expand=True)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            font=('Courier', 11),
            bg='#ecf0f1',
            fg='#2c3e50',
            wrap=tk.WORD,
            height=15
        )
        self.output_text.pack(fill='both', expand=True)
        
    def create_slider(self, parent, label, from_, to, resolution, default, row):
        """Crea uno slider con label"""
        frame = tk.Frame(parent, bg='#ffffff')
        frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        
        tk.Label(
            frame,
            text=label,
            font=('Arial', 10),
            bg='#ffffff',
            width=25,
            anchor='w'
        ).pack(side='left')
        
        var = tk.DoubleVar(value=default)
        slider = tk.Scale(
            frame,
            from_=from_,
            to=to,
            resolution=resolution,
            orient='horizontal',
            variable=var,
            bg='#ffffff',
            length=300
        )
        slider.pack(side='left', fill='x', expand=True)
        
        # Salva riferimento
        if row == 0:
            self.similarita_var = var
        elif row == 1:
            self.percentuale_var = var
        elif row == 2:
            self.sim_concettuale_var = var
            
    def create_dropdown(self, parent, label, values, default, row):
        """Crea un dropdown con label"""
        tk.Label(
            parent,
            text=label,
            font=('Arial', 10),
            bg='#ffffff',
            width=25,
            anchor='w'
        ).grid(row=row, column=0, sticky='w', pady=5)
        
        var = tk.StringVar(value=default)
        dropdown = ttk.Combobox(
            parent,
            textvariable=var,
            values=values,
            state='readonly',
            width=20
        )
        dropdown.grid(row=row, column=1, sticky='ew', pady=5)
        
        # Salva riferimento
        label_key = label.split(':')[0].lower().replace(' ', '_')
        setattr(self, f'{label_key}_var', var)
        
    def analyze(self):
        """Esegue l'analisi con il sistema esperto"""
        try:
            # Raccogli input
            similarita = self.similarita_var.get()
            percentuale = int(self.percentuale_var.get())
            sim_concettuale = self.sim_concettuale_var.get()
            citazione = self.citazione_presente_var.get()
            match_esatto = self.match_esatto_var.get()
            fonti_multiple = self.fonti_multiple_var.get()
            qualita_parafrasi = self.qualità_parafrasi_var.get()
            fonte_dichiarata = self.fonte_dichiarata_var.get()
            idea_originale = self.idea_originale_var.get()
            disciplina = self.disciplina_var.get()
            livello = self.livello_studio_var.get()
            
            # Query Prolog
            query = f"""
            analizza_testo({similarita}, {citazione}, {match_esatto}, {fonti_multiple},
                          {qualita_parafrasi}, {sim_concettuale}, {fonte_dichiarata},
                          {idea_originale}, {percentuale}, {disciplina}, {livello},
                          TipoPlagio, Gravita, Raccomandazione, RaccDisciplina)
            """
            
            results = list(self.prolog.query(query))
            
            # Mostra risultati
            self.display_results(results, similarita, percentuale)
            
        except Exception as e:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"❌ ERRORE durante l'analisi:\n\n{str(e)}")
            
    def display_results(self, results, similarita, percentuale):
        """Mostra i risultati nell'area di output"""
        self.output_text.delete(1.0, tk.END)
        
        if not results:
            self.output_text.insert(tk.END, "⚠️ Nessun risultato trovato.\n")
            self.output_text.insert(tk.END, "Verifica i parametri di input.")
            return
        
        result = results[0]
        
        # Header
        self.output_text.insert(tk.END, "=" * 80 + "\n")
        self.output_text.insert(tk.END, "📊 RISULTATI ANALISI SISTEMA ESPERTO\n")
        self.output_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Tipo plagio
        tipo = result['TipoPlagio']
        gravita = result['Gravita']
        
        self.output_text.insert(tk.END, "🔍 DIAGNOSI:\n")
        self.output_text.insert(tk.END, f"   Tipo di Plagio: {tipo.upper()}\n")
        self.output_text.insert(tk.END, f"   Gravità: {gravita.upper()}\n\n")
        
        # Confidence
        confidence = int(similarita * 100)
        self.output_text.insert(tk.END, f"📈 Confidence Score: {confidence}%\n")
        self.output_text.insert(tk.END, f"📊 Percentuale copiata: {percentuale}%\n\n")
        
        # Priorità
        if gravita == 'alta':
            priorita = "🔴 URGENTE"
        elif gravita == 'media':
            priorita = "🟡 MODERATA"
        elif gravita == 'bassa':
            priorita = "🟢 NORMALE"
        else:
            priorita = "⚪ NESSUNA"
        
        self.output_text.insert(tk.END, f"⚠️ Priorità intervento: {priorita}\n\n")
        
        # Raccomandazioni
        self.output_text.insert(tk.END, "=" * 80 + "\n")
        self.output_text.insert(tk.END, "💡 RACCOMANDAZIONI:\n")
        self.output_text.insert(tk.END, "=" * 80 + "\n\n")
        
        racc = result['Raccomandazione']
        self.output_text.insert(tk.END, f"📝 Azione generale:\n{racc}\n\n")
        
        racc_disc = result['RaccDisciplina']
        self.output_text.insert(tk.END, f"📚 Specifica disciplina:\n{racc_disc}\n\n")
        
        # Footer
        self.output_text.insert(tk.END, "=" * 80 + "\n")
        self.output_text.insert(tk.END, "✅ Analisi completata con successo!\n")
        self.output_text.insert(tk.END, "=" * 80 + "\n")

def main():
    """Funzione principale"""
    print("="*80)
    print("🚀 AVVIO SISTEMA ESPERTO PLAGIARISM DETECTION")
    print("="*80)
    
    root = tk.Tk()
    app = PlagiarismExpertSystem(root)
    
    print("\n✅ Interfaccia grafica caricata!")
    print("📋 Inserisci i parametri e premi 'ANALIZZA'")
    print("="*80 + "\n")
    
    root.mainloop()

if __name__ == "__main__":
    main()