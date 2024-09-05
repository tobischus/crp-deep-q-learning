# Deep Q-Learning für das Container Relocation Problem (CRP)

Dieses Projekt implementiert einen Deep Q-Learning (DQL) Algorithmus zur Lösung des **Container Relocation Problems (CRP)**. Ziel ist es, die Anzahl der Container-Umschichtungen zu minimieren, die notwendig sind, um einen bestimmten Container aus einem Containerlager zu entnehmen.

## Übersicht

Beim Container Relocation Problem werden Container in vertikalen Stapeln in einem Containerlager gestapelt. Die Herausforderung besteht darin, Container mit möglichst wenigen Umschichtungen zu entnehmen, da die Container in der Regel auf nicht-sequentielle Weise eingelagert werden. Blockierende Container müssen umgelagert werden, um den Zielcontainer zu erreichen.

In dieser Arbeit wird ein **Deep Q-Learning Agent** trainiert, um das optimale Umschichtungsverhalten zu erlernen. Der Agent wird durch Erfahrung (Reinforcement Learning) darauf trainiert, die Anzahl der notwendigen Umschichtungen zu minimieren.

## Technologien

- **Python 3.9.13**
- **PyTorch 2.0.0+cpu** für die Implementierung des neuronalen Netzes
- **NumPy** für numerische Berechnungen
- **Pandas** für Datenmanipulation und Analyse
- **Matplotlib** zur Visualisierung der Ergebnisse

## Algorithmus

### Zustandsdarstellung

Der Zustand wird als Liste von Stapeln repräsentiert, wobei jeder Stapel eine Liste von Containern ist. Jeder Container hat eine eindeutige Nummer. Der Agent "sieht" den aktuellen Zustand der Stapel und entscheidet, welcher Container von einem Stapel auf einen anderen verschoben werden soll.

### Aktionsraum

Die möglichen Aktionen des Agenten bestehen darin, einen Container von einem Stapel auf einen anderen zu bewegen. Nur Container, die den Zielcontainer blockieren, dürfen bewegt werden.

### Belohnungsfunktion

Der Agent erhält eine **negative Belohnung** für jede Umschichtung und eine **positive Belohnung**, wenn der Zielcontainer erfolgreich entfernt wird. Die Belohnungsfunktion motiviert den Agenten, die Anzahl der Umschichtungen zu minimieren.

### Training des Agenten

Der Agent verwendet einen **Deep Q-Network (DQN)** Ansatz. Dabei wird ein neuronales Netz trainiert, das für jede mögliche Aktion den erwarteten zukünftigen Belohnungswert (Q-Wert) vorhersagt. Das Netz wird regelmäßig mit neuen Erfahrungen aktualisiert, um das Verhalten des Agenten zu verbessern.

### Exploration vs. Exploitation

Der Agent verwendet eine **epsilon-greedy Strategie**, bei der er zu Beginn des Trainings zufällig agiert (exploration), um neue Zustände und Aktionen kennenzulernen. Im Laufe des Trainings wird die Wahrscheinlichkeit für zufällige Aktionen immer geringer, und der Agent beginnt, auf Basis seiner Erfahrungen zu handeln (exploitation).

### Zielnetzwerk

Ein zusätzliches **Zielnetzwerk** wird verwendet, um die Stabilität des Trainings zu gewährleisten. Das Zielnetzwerk wird regelmäßig mit den Gewichten des aktuellen Netzwerks aktualisiert, um verzerrte Updates zu vermeiden.

## Ergebnisse

Nach 10.000-20.000 Trainingsepisoden zeigt der Agent eine signifikante Verbesserung in der Lösung des CRP. Die durchschnittliche Anzahl der Umschichtungen sinkt, und der Agent lernt, das Containerlager effizienter zu verwalten. Die Grafiken zeigen dabei die Ergebnisse für die unterschiedlichen Layout-Größen. 

## Autor

**Tobias Schuster**  
  Technische Universität Darmstadt, Fachbereich Rechts- und Wirtschaftswissenschaften  
  E-Mail: [tobias.schuster@stud.tu-darmstadt.de](mailto:tobias.schuster@stud.tu-darmstadt.de)

Das Projekt wurde im Rahmen der **Bachelorarbeit** bei **Prof. Dr. Felix Weidinger** angefertigt und am 21.08.2024 im **Fachbereich Rechts- und Wirtschaftswissenschaften** der **Technischen Universität Darmstadt** im Fachgebiet **Management Science/Operations Research** eingereicht.

