import json
import subprocess
import time

with open("key.json", "r") as f:
    key = json.load(f)

def get_key(key_name):
    return key[key_name]

    
class ActionManager:
    def __init__(self):
        self.current_speech_process = None
        self.last_action_time = 0
        self.action_cooldown = 4  # Temps minimum entre deux actions en secondes
        
        # Définition des actions
        self.actions = {
            # Format: (right_gesture, left_gesture): (action_function, description)
            ("Right_key", "Left_key"): (self._speak_gang_pride, "Gang Pride"),
            ("Left_key", "Right_key"): (self._speak_gang_pride, "Gang Pride"),
            ("Right_peace", "Left_peace"): (self._speak_peace, "Peace"),
            ("Right_rock", "Left_rock"): (self._speak_rock, "Rock"),
            # Ajoutez d'autres actions ici
        }

    def _speak(self, text):
        """Utilise la commande say de macOS pour parler de manière asynchrone."""
        # Arrêter le processus de parole précédent s'il existe
        if self.current_speech_process is not None:
            try:
                self.current_speech_process.terminate()
            except:
                pass
        
        # Démarrer un nouveau processus de parole
        self.current_speech_process = subprocess.Popen(['say', text])

    def _speak_gang_pride(self):
        """Action: Parle de la fierté du gang."""
        self._speak("Bravo mon boy, ton gang est fier de toi")

    def _speak_peace(self):
        """Action: Parle de paix."""
        self._speak("Pisse ande love, mon frère")

    def _speak_rock(self):
        """Action: Parle de rock."""
        self._speak("Rock on, mon pote")

    def add_action(self, right_gesture, left_gesture, action_function, description):
        """Ajoute une nouvelle action au gestionnaire."""
        self.actions[(right_gesture, left_gesture)] = (action_function, description)

    def execute_action(self, right_gesture, left_gesture):
        """Exécute l'action correspondant aux gestes détectés."""
        current_time = time.time()
        
        # Vérifier si une action correspond aux gestes
        action_key = (right_gesture, left_gesture)
        if action_key in self.actions and (current_time - self.last_action_time) > self.action_cooldown:
            action_function, description = self.actions[action_key]
            print(f"Exécution de l'action: {description}")
            action_function()
            self.last_action_time = current_time

    def cleanup(self):
        """Nettoie les ressources utilisées."""
        if self.current_speech_process is not None:
            try:
                self.current_speech_process.terminate()
            except:
                pass

# Exemple d'utilisation:
# action_manager = ActionManager()
# action_manager.execute_action("Right_key", "Left_key")
    