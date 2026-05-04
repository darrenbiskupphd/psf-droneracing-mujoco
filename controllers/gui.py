import tkinter as tk
import numpy as np

class DroneGUI:
    def __init__(self, input_shaper):
        self.shaper = input_shaper
        
        self.root = tk.Tk()
        self.root.title("Drone Control Panel")
        self.root.geometry("450x350") # Bumped height slightly to fit everything nicely
        self.root.configure(bg="#2b2b2b")
        
        # Setup key bindings
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        
        # Make the window stay on top
        self.root.attributes("-topmost", True)
        
        # Main Layout Frame - Centered
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(expand=True, pady=10)
        
        # --- Row Frames for Staggered Keyboard Layout ---
        # Using anchor="w" (West/Left) and padx to create the physical offset look
        
        row1 = tk.Frame(main_frame, bg="#2b2b2b")
        row1.pack(anchor="w", padx=(30, 0), pady=3) # Standard QWE starting position
        
        row2 = tk.Frame(main_frame, bg="#2b2b2b")
        row2.pack(anchor="w", padx=(45, 0), pady=3) # ASD staggered slightly to the right
        
        row3 = tk.Frame(main_frame, bg="#2b2b2b")
        row3.pack(anchor="w", padx=(0, 0), pady=3)  # Shift starts furthest to the left
        
        # Build UI Buttons
        # Row 1: Q, W, E
        self._create_button(row1, "Q\n(+Yaw)", 'q')
        self._create_button(row1, "W\n(+Pitch)", 'w')
        self._create_button(row1, "E\n(-Yaw)", 'e')
        
        # Row 2: A, S, D, F
        self._create_button(row2, "A\n(+Roll)", 'a')
        self._create_button(row2, "S\n(-Pitch)", 's')
        self._create_button(row2, "D\n(-Roll)", 'd')
        self._create_button(row2, "F\n(Psf)", 'f')
        self.btn_f = row2.winfo_children()[-1] # Reference to change color later if needed
        
        # Row 3: Shift, Space
        self._create_button(row3, "Shift\n(-Thrust)", 'shift', width=12) # A bit long
        self._create_button(row3, "Space\n(+Thrust)", 'space', width=20) # Very long
        
        row4 = tk.Frame(main_frame, bg="#2b2b2b")
        row4.pack(anchor="w", padx=(15, 0), pady=10)
        
        # Row 4: C (Camera Toggle)
        self._create_button(row4, "C\n(Camera)", 'c', width=12)
        
        # Add instruction label at the bottom center
        instruction_text = (
            "Focus this window to fly using W/A/S/D/Q/E/Space/Shift\n"
            "or click the buttons directly to control."
        )
        self.info_label = tk.Label(
            self.root, 
            text=instruction_text, 
            bg="#2b2b2b", 
            fg="#cccccc", 
            font=("Arial", 14, "italic"), # Set to 14 as requested
            justify=tk.CENTER
        )
        self.info_label.pack(side=tk.BOTTOM, pady=(0, 20))

    def _create_button(self, parent, text, key, width=8):
        # We removed row/col arguments and are now packing them side-by-side inside their row frames
        btn = tk.Button(parent, text=text, width=width, height=2, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn.pack(side=tk.LEFT, padx=3) 
        # Bind mouse events to simulate key events
        btn.bind("<ButtonPress-1>", lambda e: self._set_key(key, True))
        btn.bind("<ButtonRelease-1>", lambda e: self._set_key(key, False))
        
    def on_key_press(self, event):
        key = event.keysym.lower()
        if key == 'space': key = 'space'
        elif key == 'shift_l' or key == 'shift_r': key = 'shift'
        self._set_key(key, True)
            
    def on_key_release(self, event):
        key = event.keysym.lower()
        if key == 'space': key = 'space'
        elif key == 'shift_l' or key == 'shift_r': key = 'shift'
        self._set_key(key, False)
        
    def _set_key(self, key, state):
        self.shaper.keys[key] = state

    def update(self):
        """Non-blocking update for the tkinter loop"""
        self.root.update_idletasks()
        self.root.update()
        
    def is_running(self):
        try:
            return self.root.winfo_exists()
        except:
            return False