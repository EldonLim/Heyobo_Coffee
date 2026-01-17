import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import os

class HeyoBo:
    """
    Animated character widget with states (idle, listening, speaking)
    and typewriter text effects.
    
    Usage:
        # Standalone window
        heyobo = HeyoBo()
        heyobo.run()
        
        # Embedded in existing Tkinter app
        root = tk.Tk()
        heyobo = HeyoBo(parent=root, assets_path="path/to/assets")
        heyobo.pack(expand=True, fill="both")
        root.mainloop()
    """
    
    # Default image dimensions
    DEFAULT_WIDTH = 1120
    DEFAULT_HEIGHT = 928
    DEFAULT_SCALE = 0.6
    
    def __init__(self, parent=None, assets_path="assets", 
                 width=None, height=None, scale=None):
        """
        Initialize HeyoBo widget.
        
        Args:
            parent: Parent Tkinter widget. If None, creates its own Tk root.
            assets_path: Path to assets folder containing states/ and transitions/
            width: Custom width (overrides scale)
            height: Custom height (overrides scale)
            scale: Scale factor for default size (default 0.6)
        """
        self._owns_root = parent is None
        self.root = parent if parent else tk.Tk()
        self.assets_path = assets_path
        
        # Calculate dimensions
        self.original_width = self.DEFAULT_WIDTH
        self.original_height = self.DEFAULT_HEIGHT
        self.aspect_ratio = self.original_width / self.original_height
        
        scale = scale or self.DEFAULT_SCALE
        self.img_width = width or int(self.original_width * scale)
        self.img_height = height or int(self.original_height * scale)
        
        # Configure root window if we own it
        if self._owns_root:
            self.root.title("HeyoBo")
            # Window size
            self.root.geometry(f"{self.img_width}x{self.img_height}")
            self.root.configure(bg="black")
            self.root.resizable(True, True)
            self.root.minsize(400, int(400 / self.aspect_ratio))

            # ---- CENTER WINDOW ----
            self.root.update_idletasks()  # ensure correct window size

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            x = (screen_width - self.img_width) // 2
            y = (screen_height - self.img_height) // 2

            self.root.geometry(f"{self.img_width}x{self.img_height}+{x}+{y}")

        # Create canvas
        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        if self._owns_root:
            self.canvas.pack(expand=True, fill="both")
        
        # State variables
        self.current_display_image = None
        self.current_text = ""
        self.current_state = "idle"
        self.is_playing_animation = False
        
        # Typewriter effect variables
        self._typewriter_text = ""
        self._typewriter_index = 0
        self._typewriter_job = None
        
        # Load idle image
        self.idle_image = self._load_png(
            os.path.join(self.assets_path, "states", "look_center.png")
        )
        
        # Bind resize event
        self.root.bind("<Configure>", self._on_resize)
        
        # Show initial image
        if self.idle_image:
            self._show_image(self.idle_image)
    
    # --------------------
    # Widget methods (for embedding)
    # --------------------
    def pack(self, **kwargs):
        """Pack the canvas widget."""
        self.canvas.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the canvas widget."""
        self.canvas.grid(**kwargs)
    
    def place(self, **kwargs):
        """Place the canvas widget."""
        self.canvas.place(**kwargs)
    
    def get_widget(self):
        """Return the canvas widget for custom layout management."""
        return self.canvas
    
    def run(self):
        """Start the main event loop (only if HeyoBo owns the root)."""
        if self._owns_root:
            self.root.mainloop()
        else:
            print("Warning: run() called but HeyoBo doesn't own the root window.")
    
    # --------------------
    # Window resize handler
    # --------------------
    def _on_resize(self, event=None):
        """Called when window is resized - maintains aspect ratio."""
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        if window_width <= 1 or window_height <= 1:
            return
        
        # Calculate size that maintains aspect ratio
        if window_width / window_height > self.aspect_ratio:
            self.img_height = window_height
            self.img_width = int(self.img_height * self.aspect_ratio)
        else:
            self.img_width = window_width
            self.img_height = int(self.img_width / self.aspect_ratio)
        
        # Redraw current image at new size
        if self.current_display_image:
            self._show_image(self.current_display_image, force_redraw=True)
    
    # --------------------
    # Image helpers
    # --------------------
    def _load_png(self, path):
        """Load original image without resizing."""
        try:
            img = Image.open(path).convert("RGBA")
            return img
        except Exception as e:
            print(f"✗ FAILED to load {path}: {e}")
            return None
    
    def _resize_image(self, img):
        """Resize image to current window size."""
        resized = img.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized)
    
    def _show_image(self, img, force_redraw=False):
        """Display image on canvas."""
        if img is None:
            print("✗ Cannot show image - it's None!")
            return
        
        if not force_redraw:
            self.current_display_image = img
        
        self.canvas.delete("all")
        
        photo = self._resize_image(img)
        self.canvas.create_image(
            self.img_width // 2, self.img_height // 2,
            image=photo
        )
        self.canvas._photo = photo  # Keep reference
        
        self._draw_text_box()
    
    def _draw_text_box(self):
        """Draw text overlay with word wrapping for long text."""
        box_height = max(60, int(self.img_height * 0.15))
        box_y = self.img_height - box_height
        font_size = max(12, int(self.img_height * 0.03))
        padding = 10
        
        self.canvas.create_rectangle(
            0, box_y,
            self.img_width, self.img_height,
            fill="#FFFFF7",
            outline="",
            width=0,
            tags="text_box"
        )
        
        # Wrap text to fit within the box width
        wrapped_text = self._wrap_text(self.current_text, font_size, self.img_width - padding * 2)
        
        self.canvas.create_text(
            self.img_width // 2,
            box_y + box_height // 2,
            text=wrapped_text,
            fill="black",
            font=("Arial", font_size),
            width=self.img_width - padding * 2,
            justify="center",
            tags="display_text"
        )
    
    def _wrap_text(self, text, font_size, max_width):
        """Wrap text to fit within max_width pixels."""
        if not text:
            return ""
        
        # Estimate characters per line based on font size and width
        # Average character width is roughly 0.6 * font_size
        char_width = font_size * 0.6
        chars_per_line = max(10, int(max_width / char_width))
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + (1 if current_line else 0) <= chars_per_line:
                current_line.append(word)
                current_length += word_length + (1 if len(current_line) > 1 else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    # --------------------
    # Typewriter effect
    # --------------------
    def _typewriter_effect(self, full_text, speed=50, on_complete=None):
        """Animate text appearing character by character."""
        if self._typewriter_job:
            self.root.after_cancel(self._typewriter_job)
        
        self._typewriter_text = full_text
        self._typewriter_index = 0
        self.current_text = ""
        
        def type_next_char():
            if self._typewriter_index < len(self._typewriter_text):
                self.current_text = self._typewriter_text[:self._typewriter_index + 1]
                self._draw_text_box()
                self._typewriter_index += 1
                self._typewriter_job = self.root.after(speed, type_next_char)
            else:
                self._typewriter_job = None
                if on_complete:
                    on_complete()
        
        type_next_char()
    
    def update_text(self, new_text, animated=True, speed=50):
        """Update displayed text with optional typewriter effect."""
        if self._typewriter_job:
            self.root.after_cancel(self._typewriter_job)
            self._typewriter_job = None
        
        if animated and new_text:
            self._typewriter_effect(new_text, speed)
        else:
            self.current_text = new_text
            self._draw_text_box()
    
    # --------------------
    # Animation playback
    # --------------------
    def play_animation(self, gif_path, on_complete=None):
        """Play a GIF animation."""
        if self.is_playing_animation:
            print("⚠️ Animation already playing, skipping...")
            return
        
        self.is_playing_animation = True
        print(f"Playing animation: {gif_path}")
        
        try:
            gif = Image.open(gif_path)
            frames_original = []
            
            for frame in ImageSequence.Iterator(gif):
                frames_original.append(frame.copy().convert("RGBA"))
            
            delay = gif.info.get("duration", 40)
            
            def play_frame(index=0):
                if index < len(frames_original):
                    self._show_image(frames_original[index])
                    self.root.after(delay, play_frame, index + 1)
                else:
                    self.is_playing_animation = False
                    self._show_image(self.idle_image)
                    print("Animation complete, returned to idle")
                    if on_complete:
                        on_complete()
            
            play_frame()
        except Exception as e:
            print(f"✗ Failed to load animation {gif_path}: {e}")
            self.is_playing_animation = False
            self._show_image(self.idle_image)
            if on_complete:
                on_complete()
    
    # --------------------
    # State actions (public API)
    # --------------------
    def idle(self):
        """Transition to idle state."""
        self.current_state = "idle"
        self.update_text("", animated=False)
        
        anim_path = os.path.join(self.assets_path, "transitions", "idle_animation.gif")
        if os.path.exists(anim_path):
            self.play_animation(anim_path)
        else:
            print("⚠️ idle_animation.gif not found")
            self._show_image(self.idle_image)
    
    def listen(self, text="Listening..."):
        """Transition to listening state."""
        self.current_state = "listening"
        self.update_text(text, animated=True, speed=50)
        
        anim_path = os.path.join(self.assets_path, "transitions", "listening_animation.gif")
        if os.path.exists(anim_path):
            self.play_animation(anim_path)
        else:
            print("⚠️ listening_animation.gif not found")
            self._show_image(self.idle_image)
    
    def speak(self, text="Speaking...", typewriter=True, speed=30):
        """Transition to speaking state."""
        self.current_state = "speaking"
        self.update_text(text, animated=typewriter, speed=speed)
        
        anim_path = os.path.join(self.assets_path, "transitions", "speaking_animation.gif")
        if os.path.exists(anim_path):
            self.play_animation(anim_path)
        else:
            print("⚠️ speaking_animation.gif not found")
            self._show_image(self.idle_image)
    
    def get_state(self):
        """Return the current state."""
        return self.current_state
    
    def is_animating(self):
        """Return True if an animation is currently playing."""
        return self.is_playing_animation
    
    # --------------------
    # Cleanup
    # --------------------
    def destroy(self):
        """Clean up resources."""
        if self._typewriter_job:
            self.root.after_cancel(self._typewriter_job)
        if self._owns_root:
            self.root.destroy()
