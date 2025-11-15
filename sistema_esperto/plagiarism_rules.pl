/*
================================================================================
PLAGIARISM DETECTION - KNOWLEDGE BASE SYSTEM
Knowledge Base complessa per classificazione e reasoning sul plagio accademico
================================================================================
Autore: [Raffaella Nitti]
Matricola: [796132]
*/

/* ============================================================================
   FATTI BASE - Definizioni fondamentali
   ============================================================================ */

% Tipi di plagio riconosciuti
tipo_plagio(letterale).
tipo_plagio(mosaico).
tipo_plagio(parafrasi_insufficiente).
tipo_plagio(idee).
tipo_plagio(nessuno).

% Livelli di gravità
gravita(alta).
gravita(media).
gravita(bassa).
gravita(nessuna).

% Discipline accademiche con regole diverse
disciplina(informatica).
disciplina(medicina).
disciplina(filosofia).
disciplina(ingegneria).
disciplina(letteratura).

% Livelli di studio
livello_studio(bachelor).
livello_studio(master).
livello_studio(phd).

/* ============================================================================
   REGOLE BASE - Classificazione tipo plagio
   ============================================================================ */

% Plagio Letterale: alta similarità + nessuna citazione + match esatto
classifica_plagio_letterale(Similarita, Citazione, MatchEsatto, letterale) :-
    Similarita > 0.8,
    Citazione = no,
    MatchEsatto = yes.

% Plagio Mosaico: media similarità + fonti multiple + no citazioni
classifica_plagio_mosaico(Similarita, FontiMultiple, Citazione, mosaico) :-
    Similarita > 0.5,
    Similarita =< 0.8,
    FontiMultiple = yes,
    Citazione = no.

% Parafrasi Insufficiente: alta similarità + parafrasi scarsa + fonte dichiarata
classifica_parafrasi_insufficiente(Similarita, QualitaParafrasi, FonteDichiarata, parafrasi_insufficiente) :-
    Similarita > 0.7,
    QualitaParafrasi = scarsa,
    FonteDichiarata = yes.

% Plagio di Idee: similarità concettuale + no citazione + idee non originali
classifica_plagio_idee(SimilaritaConcettuale, Citazione, IdeaOriginale, idee) :-
    SimilaritaConcettuale > 0.6,
    Citazione = no,
    IdeaOriginale = no.

% Nessun Plagio: bassa similarità o citazioni corrette
classifica_nessun_plagio(Similarita, Citazione, nessuno) :-
    (Similarita =< 0.3 ; Citazione = corretta).

/* ============================================================================
   REGOLE AVANZATE - Valutazione gravità contestuale
   ============================================================================ */

% Valuta gravità considerando contesto
valuta_gravita(TipoPlagio, Percentuale, Disciplina, LivelloStudio, Gravita) :-
    gravita_base(TipoPlagio, Percentuale, GravitaBase),
    fattore_disciplina(Disciplina, FattoreDisciplina),
    fattore_livello(LivelloStudio, FattoreLivello),
    applica_fattori(GravitaBase, FattoreDisciplina, FattoreLivello, Gravita).

% Gravità base per tipo di plagio
gravita_base(letterale, Perc, alta) :- Perc > 30.
gravita_base(letterale, Perc, media) :- Perc > 10, Perc =< 30.
gravita_base(letterale, Perc, bassa) :- Perc =< 10.

gravita_base(mosaico, Perc, alta) :- Perc > 40.
gravita_base(mosaico, Perc, media) :- Perc > 20, Perc =< 40.
gravita_base(mosaico, Perc, bassa) :- Perc =< 20.

gravita_base(parafrasi_insufficiente, _, media).
gravita_base(idee, _, media).
gravita_base(nessuno, _, nessuna).

% Fattori contestuali - Disciplina
fattore_disciplina(medicina, rigoroso) :- 
    % Medicina richiede citazioni rigorose per sicurezza pazienti
    !.
fattore_disciplina(ingegneria, moderato) :-
    % Ingegneria: formule standard ammesse
    !.
fattore_disciplina(filosofia, rigoroso) :-
    % Filosofia: idee originali fondamentali
    !.
fattore_disciplina(informatica, moderato).
fattore_disciplina(letteratura, rigoroso).

% Fattori contestuali - Livello studio
fattore_livello(bachelor, tollerante).
fattore_livello(master, moderato).
fattore_livello(phd, rigoroso).

% Applica fattori per gravità finale
applica_fattori(bassa, rigoroso, rigoroso, media).
applica_fattori(media, rigoroso, rigoroso, alta).
applica_fattori(alta, _, rigoroso, alta).
applica_fattori(Gravita, tollerante, tollerante, Gravita).
applica_fattori(alta, moderato, moderato, alta).
applica_fattori(media, moderato, moderato, media).
applica_fattori(bassa, _, _, bassa).
applica_fattori(nessuna, _, _, nessuna).

/* ============================================================================
   REGOLE ECCEZIONI - Casi speciali
   ============================================================================ */

% Common Knowledge: noto a tutti, non richiede citazione
common_knowledge(Concetto) :-
    member(Concetto, [
        'acqua bolle a 100 gradi',
        'terra ruota intorno al sole',
        'formula pitagorica',
        'definizione algoritmo base'
    ]).

% Formule standard: ammesse senza citazione in alcune discipline
formula_standard(Formula, Disciplina) :-
    member(Disciplina, [ingegneria, informatica, matematica]),
    membro_formule_standard(Formula).

membro_formule_standard('a^2 + b^2 = c^2').
membro_formule_standard('E = mc^2').
membro_formule_standard('F = ma').

% Verifica se è eccezione (non è plagio)
eccezione_plagio(Testo, Disciplina, yes) :-
    (common_knowledge(Testo) ; 
     formula_standard(Testo, Disciplina)).

eccezione_plagio(_, _, no).

/* ============================================================================
   REGOLE RACCOMANDAZIONI - Azioni correttive
   ============================================================================ */

% Genera raccomandazioni basate su tipo e gravità
raccomandazione(letterale, alta, 'RISCRIVERE COMPLETAMENTE il testo. Consultare fonti originali e parafrasare con parole proprie.').
raccomandazione(letterale, media, 'Riscrivere le parti copiate. Aggiungere citazioni esplicite per frasi mantenute.').
raccomandazione(letterale, bassa, 'Aggiungere citazioni per le frasi riprese. Verificare formato citazione.').

raccomandazione(mosaico, alta, 'Riscrivere integralmente. Studiare le fonti e sviluppare argomento originale.').
raccomandazione(mosaico, media, 'Ristrutturare il testo con argomentazioni proprie. Citare ogni fonte usata.').

raccomandazione(parafrasi_insufficiente, _, 'Migliorare la parafrasi: cambiare struttura frase e vocabolario. Citare comunque la fonte.').

raccomandazione(idee, _, 'Citare esplicitamente la fonte delle idee. Sviluppare contributo personale.').

raccomandazione(nessuno, nessuna, 'Nessuna azione necessaria. Lavoro corretto.').

% Raccomandazioni specifiche per disciplina
raccomandazione_disciplina(medicina, 'Usare formato Vancouver. Citare ogni affermazione clinica.').
raccomandazione_disciplina(filosofia, 'Usare formato MLA. Distinguere chiaramente idee proprie da altrui.').
raccomandazione_disciplina(ingegneria, 'Usare formato IEEE. Formule standard OK, ma citare applicazioni specifiche.').
raccomandazione_disciplina(informatica, 'Usare formato ACM/IEEE. Codice: citare librerie, non algoritmi base.').
raccomandazione_disciplina(letteratura, 'Usare formato MLA. Citare ogni riferimento testuale.').

/* ============================================================================
   REGOLE INFERENZA COMPLESSA - Reasoning multi-livello
   ============================================================================ */

% Sistema di inferenza principale
analizza_testo(Similarita, Citazione, MatchEsatto, FontiMultiple, 
               QualitaParafrasi, SimilaritaConcettuale, FonteDichiarata,
               IdeaOriginale, Percentuale, Disciplina, LivelloStudio,
               TipoPlagio, Gravita, Raccomandazione, RaccDisciplina) :-
    
    % Step 1: Verifica eccezioni
    \+ eccezione_plagio(_, Disciplina, yes),
    
    % Step 2: Classifica tipo plagio (prova tutte le regole)
    (
        classifica_plagio_letterale(Similarita, Citazione, MatchEsatto, TipoPlagio) ;
        classifica_plagio_mosaico(Similarita, FontiMultiple, Citazione, TipoPlagio) ;
        classifica_parafrasi_insufficiente(Similarita, QualitaParafrasi, FonteDichiarata, TipoPlagio) ;
        classifica_plagio_idee(SimilaritaConcettuale, Citazione, IdeaOriginale, TipoPlagio) ;
        classifica_nessun_plagio(Similarita, Citazione, TipoPlagio)
    ),
    
    % Step 3: Valuta gravità contestuale
    valuta_gravita(TipoPlagio, Percentuale, Disciplina, LivelloStudio, Gravita),
    
    % Step 4: Genera raccomandazioni
    raccomandazione(TipoPlagio, Gravita, Raccomandazione),
    raccomandazione_disciplina(Disciplina, RaccDisciplina).

% Query helper per risultati completi
diagnosi_completa(Input, Output) :-
    Input = [Sim, Cit, Match, Multi, ParQual, SimCon, FontDich, IdeaOrig, Perc, Disc, Liv],
    analizza_testo(Sim, Cit, Match, Multi, ParQual, SimCon, FontDich, IdeaOrig, Perc, Disc, Liv,
                   Tipo, Grav, Racc, RaccDisc),
    Output = [Tipo, Grav, Racc, RaccDisc].

/* ============================================================================
   REGOLE UTILITY - Helper functions
   ============================================================================ */

% Calcola confidence score
confidence_score(Similarita, MatchEsatto, Score) :-
    (MatchEsatto = yes -> BaseScore = Similarita ;
     BaseScore is Similarita * 0.8),
    Score is BaseScore * 100.

% Priorità intervento
priorita_intervento(alta, urgente).
priorita_intervento(media, moderata).
priorita_intervento(bassa, normale).
priorita_intervento(nessuna, nessuna).

/* ============================================================================
   FINE KNOWLEDGE BASE
   ============================================================================ */