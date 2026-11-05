# II2202 — Knowledge base del progetto Jev e network digital twin

Aggiornata al 6 ottobre 2026. Documento di lavoro per Jago Revrenna e Gabriele Bandino: raccoglie le decisioni attuali, le ragioni del metodo, il materiale del corso e i punti ancora aperti. **Non è un risultato sperimentale né sostituisce il research plan o la verifica delle fonti da parte degli autori.**

## 1. Quale versione del progetto vale ora

La direzione attuale è uno **studio sperimentale della sincronizzazione di un network digital twin (NDT)**. La domanda è se un modello decisionale esterno, **Jev**, possa scegliere fra due *politiche di polling* meglio di alternative semplici, tenendo conto sia dell'errore dello stato del twin sia del traffico necessario per aggiornarlo.

La distinzione decisiva è questa: **Jev sceglie quale politica governa il prossimo periodo, non quale link interrogare**. Ogni poll misura lo stesso insieme fisso di link. Il codice calcola i riassunti delle misure, invoca Jev, riceve una delle due opzioni ammesse e applica la politica selezionata. Jev non fa polling, non predice lo stato vero della rete, non vede eventi non ancora misurati e non viene addestrato per il progetto.

```text
Traccia sintetica della rete → misure ricevute dal twin → riassunto nel codice
                                              ↓
                  al review point: regola semplice OPPURE Jev
                                              ↓
                      polling fisso OPPURE change-aware
                                              ↓
             nuovi poll → consegna ritardata → stato memorizzato nel twin
```

Una precedente nota interna, [`teacher-feedback-notes.md`](II2202_Project_Proposal_Jev/teacher-feedback-notes.md), descriveva ancora Jev come selettore del **link**. È una fotografia di una fase superata. Per lo scope attuale fanno fede il [revised project proposal](II2202_Project_Proposal_Jev/II2202-proposal.tex) e il [research plan](II2202_Research_Plan_Jev/II2202-research-plan.tex).

## 2. Persone e documenti

| Persona | Ruolo nel progetto |
|---|---|
| Jago Revrenna — revrenna@kth.se | Coautore; nel piano proposto guida simulatore, updater del twin e controller convenzionali. |
| Gabriele Bandino — bandino@kth.se | Coautore; nel piano proposto guida interfaccia Jev, logging e verifica accesso/costi. |
| Nadia Bekkali | Mentor teacher; deve poter valutare fattibilità e metodo. |

La divisione del lavoro è **proposta, non un impegno confermato**. Entrambi devono poter spiegare l'intero esperimento, verificare fonti e risultati, e rivedere il lavoro dell'altro.

Documenti attuali:

- [Proposta rivista, sorgente LaTeX](II2202_Project_Proposal_Jev/II2202-proposal.tex): inquadramento, domanda, contributo e metodo plausibile.
- [Research plan, sorgente LaTeX](II2202_Research_Plan_Jev/II2202-research-plan.tex): protocollo implementabile, analisi, quality assurance, tempi, rischi e uso dell'AI.
- [README del research plan](II2202_Research_Plan_Jev/README.md): sintesi dello scope e verifiche da fare.
- [Prompt originale delle slide](II2202_Video_Presentation/lovable_plan_prompt.md): utile come storico; le modifiche successive nel progetto Lovable hanno priorità per l'aspetto attuale delle slide.
- [Presentazione interattiva Research su Lovable](https://lovable.dev/projects/32a40ecf-2a56-47a9-9119-967a97a08817): progetto delle cinque slide animate. Non risulta pubblicato.

Documenti storici da **non** trattare come protocollo corrente: il [riferimento generale del 15 settembre](II2202_full_course_and_project_reference_2026-09-15.docx), il [controllo citazioni della precedente review](II2202_citation_check_2026-09-15.md) e la nota interna sul primo scope Jev/link. Sono utili per capire come si è arrivati alla decisione odierna, ma contengono direzioni poi abbandonate o date ormai da riconfermare.

## 3. Problema, domande e contributo

Il twin conserva per ogni link l'**ultima utilizzazione ricevuta**. Se la rete cambia fra due osservazioni, il valore memorizzato può diventare inesatto. Polling più frequente tende a ridurre il tempo in cui si conserva un valore vecchio, ma usa più messaggi. Un valore vecchio, però, può essere ancora corretto se il link è rimasto stabile: *età dell'informazione* ed *errore dello stato* non sono la stessa misura.

**Domanda principale.** In quali condizioni di traffico, se ve ne sono, la selezione di Jev fra polling a intervallo fisso e polling change-aware riduce l'errore del twin a traffico di monitoraggio comparabile, rispetto alle due politiche usate sempre e a uno switch deterministico?

**Domanda di supporto.** In che misura il ritardo di consegna delle misure, il tempo di risposta di Jev e i byte scambiati con la sua API riducono o annullano un eventuale vantaggio?

**Ipotesi di lavoro, non risultato.** Lo switch potrebbe aiutare quando periodi quieti e variabili si alternano. Una regola semplice, oppure la sola politica change-aware, potrebbe però fare altrettanto bene o meglio. Su traffico stabile Jev potrebbe aggiungere solo overhead.

**Contributo atteso.** Non una sintesi di paper: un confronto riproducibile, per scenario, di quattro controller sulle **stesse tracce**, con grafici errore–traffico, differenze a livelli di comunicazione sovrapposti e costo aggiuntivo di Jev. Anche l'esito «la regola semplice basta» sarebbe un risultato utile e da riportare.

## 4. Jev nel progetto, senza ambiguità

Nel design proposto Jev è il modello decisionale di TypeSafe raggiunto tramite API. L'interfaccia **Choice** consente di chiedere una scelta fra opzioni definite: qui le sole opzioni sono `fixed` e `change-aware`. Il sistema locale prepara un input strutturato usando **soltanto misure già ricevute**: valori e timestamp, cambiamenti osservati, età delle misure e conteggio recente dei poll. Una regola deterministica riceve le stesse *tipologie* di informazione. Le storie effettivamente osservate possono differire, perché i controller prendono decisioni precedenti diverse.

Esempio **ipotetico**, da usare per spiegare il meccanismo e non come evidenza di una capacità provata: se le misure ricevute sono state quiete e poi variabili, al prossimo review point il codice riassume questa storia. Jev potrebbe rispondere `change-aware`; da quel momento il codice usa tale politica. Jev non ha visto l'evento corrente non misurato né il futuro. La scelta arriva dopo un tempo di risposta misurabile.

Confini importanti:

- Jev non determina un nuovo intervallo numerico e non sceglie il link da interrogare.
- Non confrontiamo Jev con un generico chatbot/LLM: il test riguarda questa scelta fra due opzioni e il suo costo operativo.
- Non è previsto training o fine-tuning del modello.
- Una probabilità/confidenza restituita dall'interfaccia non dimostra che lo stato del twin sia corretto.
- Documentazione di prodotto = prova delle caratteristiche dell'interfaccia, **non** prova che Jev migliori la sincronizzazione.
- Accesso, versione, limiti, quote e prezzo effettivo vanno verificati **prima** dei main runs. Non assumere che Jev sia gratuito.

## 5. Protocollo sperimentale attualmente previsto

### Ambiente e controller

Una simulazione Python riproduce domande sintetiche su un piccolo grafo fisso, con percorsi condivisi e circa otto link monitorati come **stima di partenza**. Il simulatore conosce la vera utilizzazione per valutare l'errore. I controller vedono solo le misure già arrivate al twin. A ogni poll viene campionato l'intero insieme fisso di link; la misura raggiunge il twin dopo il ritardo impostato per lo scenario. Il twin conserva l'ultimo valore ricevuto e rifiuta un aggiornamento più vecchio di quello già accettato.

| Controller | Funzione |
|---|---|
| Fixed | Poll a intervallo costante, senza switch. |
| Change-aware | Accorcia o allunga l'intervallo entro limiti fissati, usando cambiamenti già osservati. |
| Rule switch | A review point prefissati sceglie fra le stesse due politiche con soglie deterministiche. |
| Jev switch | Agli stessi review point chiede a Jev di scegliere fra le stesse due politiche. |

Le implementazioni e impostazioni delle due politiche devono essere le stesse per controller statici e switcher. Durante l'attesa di Jev resta attiva la politica precedente; se la risposta è invalida o fuori tempo, resta la precedente e si registra un fallback. Un solo request Jev può essere in sospeso. Quando una politica cambia, si conserva la storia delle misure e si programma il poll successivo rispetto all'ultimo poll, senza crearne uno extra solo per lo switch. Nessun selector può usare misure ricevute dopo la costruzione della propria richiesta.

### Condizioni, pilot e freeze

Le condizioni pianificate sono traffico quasi stabile, burst brevi, alternanza quiete/variabilità e spostamento persistente del carico; ognuna con ritardo di consegna basso o elevato. La topologia rimane fissa: il progetto **non** vuole fermarsi alla sola variazione del numero di nodi, ma studia la scelta di sincronizzazione sotto dinamiche e ritardi differenti.

Il pilot deve verificare il codice e fissare: ampiezze, durata, ritardi, intervalli candidati, soglie, frequenza di review, formato della richiesta Jev, timeout e numero di ripetizioni. Nel piano compaiono come **stime da verificare**, non valori finali, una traccia di cinque minuti, tre intervalli fixed candidati e circa 12–20 seed held-out per condizione. Dopo il pilot si congelano configurazione e analisi; i main runs usano seed nuovi e condivisi tra i quattro controller. Non si modifica il prompt Jev o la soglia della regola osservando i risultati held-out.

### Dati da registrare e analisi

Log previsti: stato vero e memorizzato, seed, scenario, versione di simulatore/modello, poll inviati, tempi di dispatch e arrivo, input della decisione, scelta, latenza, risposte invalide, fallback, dimensioni dei messaggi e costi API. Credenziali non nei log.

**Metriche primarie:** errore assoluto medio fra utilizzazione vera e valore memorizzato del twin, mediato su link e tempi dopo un warm-up comune, espresso in punti percentuali; byte applicativi di richiesta e risposta del monitoraggio. Il traffico del warm-up rimane nel totale dei costi. **Metriche di supporto:** numero di poll, ritardo nel rilevare cambiamenti, tempo decisionale, byte e addebiti dell'API Jev, frequenza dei fallback.

Si confrontano errori **a livelli di traffico sovrapposti**: più accuratezza ottenuta solo con più polling non prova che la decisione sia migliore. Se le fasce di traffico non si sovrappongono, si mostra il trade-off senza classifica artificiale. Byte dell'API e latenza Jev si riportano separatamente; poi i byte API si aggiungono a quelli di monitoraggio per mostrare il costo applicativo complessivo. Le coppie di run condividono il seed; l'unità statistica è il **run intero**, non ogni campione temporale. Si riportano dimensione/direzione delle differenze e intervalli d'incertezza, anche per risultati nulli o sfavorevoli.

### Quality assurance e limiti

Test previsti: stato costante, errore calcolato a mano, switch senza poll extra, risposte ritardate o fuori ordine, timeout, fallback e assenza di leakage di dati futuri. I due autori controlleranno componenti reciproci e log campione. Si conserveranno generatori, seed, configurazioni, script di analisi e input/decisioni Jev per riprodurre l'analisi anche se il servizio cambia.

La simulazione isola la decisione ma non riproduce tutti gli effetti di trasporto e delle apparecchiature reali. Un grafo, tracce sintetiche e due politiche predefinite limitano la generalizzazione. Un cambiamento imprevedibile e non ancora osservato non può essere noto a nessuno dei controller basati su osservazioni: non va presentato come un difetto esclusivo di Jev.

## 6. Fattibilità, costo e lavoro pratico

Il requisito espresso da Jago è **zero spesa personale**. Il piano include un gate iniziale: controllare accesso API, quote/crediti, versione, numero di richieste, latenza e addebito previsto. Fissare un cap che impedisca spesa di tasca propria. Se non si può fare l'esperimento così, discutere con Nadia una riduzione o revisione dello scope **prima** dei main runs; non sostituire Jev silenziosamente con un altro modello. Il fatto che il provider pubblichi prezzi per token non dimostra né disponibilità né assenza di un piano gratuito.

Finestre di lavoro indicate nel research plan, da ricontrollare nel calendario ufficiale: 6–9 ottobre feedback/accesso; 10–16 ottobre simulatore e baseline; 17–23 ottobre integrazione/pilot/freeze; 24 ottobre–7 novembre main runs; 8–20 novembre analisi e prima bozza; fine novembre–dicembre revisioni e preparazione finale. Sono **obiettivi**, non attività già concluse né proroghe ufficiali.

## 7. Feedback di Nadia e requisiti del corso

Nel feedback scritto alla prima proposta, Nadia ha chiesto tre correzioni:

1. Esplicitare il **prodotto/contributo** oltre alla lettura di paper. Lo studio sperimentale attuale risponde con un confronto a quattro controller, grafici e condizioni in cui un metodo aiuta o no.
2. Separare chiaramente **approcci**, **condizioni della rete** e **metriche**, scegliendo una o due misure primarie. Il piano distingue controller, scenari e due assi principali: errore e byte di monitoraggio.
3. Spiegare quale evidenza sostiene un giudizio di «adatto a una condizione». Il piano non usa la sola impressione tratta da un paper: prevede run ripetuti, seed accoppiati, sovrapposizione del costo di comunicazione e incertezza.

La [nota dell'audio](II2202_Project_Proposal_Jev/teacher-feedback-notes.md) è una **parafrasi da trascrizione automatica non verificata**. Indica di tenere la proposta breve, una domanda principale e una secondaria, mostrare concretamente cosa si farà e quale esito ci si aspetta, e cambiare una parte del processo invece di ridisegnare l'intero NDT. Non è una citazione letterale di Nadia né prova che abbia approvato il disegno attuale.

La *project proposal* è un compito formativo: problema specifico, lavori pertinenti, domande rispondibili, metodo plausibile e fonti effettivamente usate. La lunghezza tipica indicata dall'assegnazione è 3–5 pagine, non un limite rigido. Il *research plan* deve dettagliare disegno, dati, procedura, analisi, quality assurance, uso di AI, responsabilità, milestone e rischi, abbastanza da poterlo davvero eseguire. Il testo scientifico deve essere rielaborato e verificato dagli studenti; gli strumenti di AI possono aiutare, ma non sostituiscono la loro responsabilità. Le istruzioni nel template vanno rimosse prima della consegna.

Date dei compiti, quiz e seminario vanno sempre controllati in Canvas: il [documento generale](II2202_full_course_and_project_reference_2026-09-15.docx) è aggiornato solo al 15 settembre e non va usato come calendario definitivo.

## 8. Presentazione video, etica e sostenibilità

L'assegnazione chiede **massimo 10 minuti e massimo cinque slide** nell'ordine: 1 titolo/team, 2 tema/problema, 3 metodo/work plan, 4 etica, 5 sostenibilità. Entrambi gli studenti devono essere visibili e parlare per tempi grossomodo uguali. La versione preparata mira a **9 minuti, 4:30 a testa**, lasciando un minuto di margine. Le slide supportano il parlato con schemi, etichette e animazioni a step; le note sono per i relatori. Il video va pubblicato nella discussione della propria sezione in Canvas e il PDF delle slide caricato nell'assegnazione. Il peer feedback valuta chiarezza, qualità visiva e riflessioni specifiche su rischi, stakeholder, conseguenze e mitigazioni.

**Slide 1.** Titolo del progetto, Jago e Gabriele, data.

**Slide 2.** Il valore nel twin può restare fermo fra due poll mentre la rete cambia. Lo schema è **illustrativo, non un risultato**. Alla review programmata: `received history → Jev → fixed OR change-aware`; il codice riassume l'input ed esegue la scelta nel periodo successivo. L'esempio orale è ipotetico.

**Slide 3.** Una traccia sintetica confrontata sotto quattro controller, con errore del twin e byte di monitoraggio come assi principali, costo/tempo Jev separato e work plan con gate di fattibilità.

**Slide 4.** Catena `misure → scelta della politica → stato del twin → operatore`. Rischio futuro A: telemetria reale inviata a un'API esterna potrebbe rivelare pattern della rete; nel nostro studio si usano solo dati sintetici, senza controllo live o credenziali. Rischio futuro B: poll troppo radi o decisione tardiva potrebbero lasciare il twin obsoleto e fuorviare l'operatore; fallback, log della staleness e revisione umana sono possibili contromisure. **Non sono incidenti osservati.** La slide attuale resta un unico diagramma con zoom su A, zoom su B e ritorno alla panoramica.

**Slide 5.** Bilancia neutra: meno poll possono significare meno traffico di monitoraggio, ma Jev richiede API/compute e poll troppo radi possono peggiorare il twin. Si confrontano traffico ed errore; **energia e CO₂ non sono misurati direttamente**. La bilancia non indica che Jev abbia già vinto. Un risultato negativo, in cui basta una regola semplice, resta informativo.

La registrazione va provata a tempo reale: contenuto scritto nelle note e 4:30 a testa sono una pianificazione, non la prova che il video finale rimarrà sotto 10 minuti.

## 9. Etica, sostenibilità e integrità

Lo studio non necessita di partecipanti umani, tracce operative private o controllo di una rete reale. I rischi della slide 4 riguardano soprattutto un possibile uso futuro: esposizione di telemetria a un provider esterno e decisioni prese sulla base di uno stato obsoleto. Le contromisure previste sono dati sintetici, niente credenziali nei log, niente controllo live, timeout/fallback, tracciamento di decisioni e staleness, e valutazione umana prima di un deployment.

Più o meno messaggi **non equivale** a una misura di consumo energetico. Jev aggiunge chiamate API e calcolo esterno. Si misureranno byte applicativi e latenza, non energia o emissioni; la discussione sulla sostenibilità deve chiamare queste ultime possibili implicazioni, non risultati.

L'AI è stata usata nella stesura del piano e può essere usata per lingua, codice e scoperta delle fonti. Gli autori devono leggere le fonti, controllare codice e risultati, rivedere il testo e dichiarare l'aiuto materiale secondo le regole del corso. Nessun AI detector può garantire che un testo «non sembri AI», e non è corretto inventare lavoro, letture o risultati svolti da voi. Jev come oggetto sperimentale è distinto dall'assistenza alla scrittura.

## 10. Fonti correnti e limiti delle citazioni

Queste sono le **sette voci citate nei testi attuali**; titolo, DOI/URL e formulazione vanno ricontrollati prima di consegnare o riutilizzare il materiale. Non attribuire ai lavori risultati che non verificano.

| Fonte | Perché compare | Limite d'uso |
|---|---|
| [Zhou et al., NDT reference architecture, draft 13](https://datatracker.ietf.org/doc/html/draft-irtf-nmrg-network-digital-twin-arch-13) | Definire NDT e ruolo delle osservazioni. | Internet-Draft in lavorazione, non standard definitivo né evidenza della performance di Jev. |
| [van Adrichem et al., OpenNetMon (2014)](https://doi.org/10.1109/NOMS.2014.6838228) | Motivare una baseline che adatta il polling ai cambiamenti osservati. | Il progetto implementa una regola semplice; non riproduce il sistema per-flow del paper. |
| [Wieme et al., monitoring frequency (2025)](https://doi.org/10.1109/NOMS57970.2025.11073664) | Trattare accuratezza e overhead come trade-off. | Contesto Bluetooth Mesh, non prova generalizzabile a tutte le reti. |
| [Yates et al., Age of Information (2021)](https://doi.org/10.1109/JSAC.2021.3065072) | Distinguere età degli update ed errore dello stato. | Minimizzare l'età non garantisce sempre di minimizzare l'errore del twin. |
| [TypeSafe, Choice](https://docs.typesafe.ai/primitives/choice) | Interfaccia a scelta fra opzioni definite. | Documentazione del prodotto, non benchmark NDT. |
| [TypeSafe, Models](https://docs.typesafe.ai/models) | Versione/alias e personalizzazione. | Confermare comportamento e disponibilità al momento del pilot. |
| [TypeSafe, Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) | Cautela sulle richieste numeriche. | Limite documentato del prodotto, non dimostrazione che Jev fallirà qui. |

Il [fact check del 15 settembre](II2202_citation_check_2026-09-15.md) riguarda la **precedente proposta di review** e non certifica automaticamente la bibliografia Jev odierna. La fonte Kitchenham/Charters era stata esplicitamente esclusa: non reintrodurla. Le repository menzionate nella vecchia fase, per esempio CARLA-TWIN, non sono componenti già scelti o eseguiti nel protocollo corrente.

## 11. Decisioni ancora aperte e checklist prima dei main runs

- Confermare con Gabriele e Nadia che lo scope *policy selection fra fixed e change-aware* è quello da eseguire; non assumere un'approvazione implicita dal feedback precedente.
- Verificare documentazione Jev aggiornata, crediti/accesso disponibili, versione, timeout, prezzi e un piano realmente a **zero esborso personale**.
- Concordare la divisione dei compiti, la frequenza di review, il request cap, le soglie e i parametri del generatore nel pilot.
- Congelare codice, prompt, configurazioni, seed di sviluppo e piano di analisi **prima** dei run held-out.
- Ricontrollare ogni citazione nel contesto esatto in cui viene usata e la bibliografia finale; documentare eventuali modifiche dei prodotti online.
- Confermare in Canvas scadenze, quiz, modalità di resubmission e data del seminario; non prendere il calendario di settembre come fonte definitiva.
- Esportare il PDF delle slide, leggerlo per intero, provare il video sotto 10 minuti con entrambi visibili e verificare i passaggi di parola.

## 12. Sintesi da usare in una conversazione

> Vogliamo capire se Jev può scegliere **quando usare una politica di polling fisso e quando una adattiva** per sincronizzare lo stato di un network digital twin. Non gli facciamo scegliere link o prevedere la rete. Confrontiamo la scelta di Jev con le due politiche usate da sole e con una semplice regola di switch, sulle stesse tracce sintetiche. Il risultato sarà un confronto fra errore del twin, traffico di monitoraggio e costo/tempo dell'API. Se Jev non batte la regola semplice a costo comparabile, anche questo è un risultato. Prima degli esperimenti dobbiamo confermare accesso e costo senza spendere di tasca nostra.
