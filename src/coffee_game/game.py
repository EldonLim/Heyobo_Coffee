"""Coffee Catcher Game - Class-based implementation for easy integration."""

import os
import pygame
import random
import time
from typing import Optional


class CoffeeGame:
    """A coffee catching game where players fill their cup while avoiding bombs."""

    # Colors
    WHITE = (255, 255, 255)
    BROWN = (139, 69, 19)
    CREAM = (255, 253, 208)
    RED = (255, 0, 0)
    DARK_BROWN = (101, 67, 33)
    GREEN = (0, 200, 0)
    LIGHT_BROWN = (210, 180, 140)

    def __init__(
        self,
        width: int = 600,
        height: int = 800,
        game_time: int = 30,
        spawn_interval: int = 1400,
        bomb_chance: float = 0.1,
        screen: Optional[pygame.Surface] = None,
        use_hand_control: bool = False,
    ):
        """
        Initialize the Coffee Game.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            game_time: Time limit in seconds
            spawn_interval: Milliseconds between object spawns
            bomb_chance: Probability (0-1) of spawning a bomb vs coffee
            screen: Optional existing pygame screen to use
            use_hand_control: Enable hand gesture control via webcam
        """
        self.width = width
        self.height = height
        self.game_time = game_time
        self.spawn_interval = spawn_interval
        self.bomb_chance = bomb_chance

        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
        
        pygame.mixer.init()

        # Screen setup
        if screen is not None:
            self.screen = screen
            self._owns_screen = False
        else:
            # Position window on the right side of the screen
            screen_info = pygame.display.Info()
            screen_width = screen_info.current_w
            x_pos = screen_width - width - 50  # 50px padding from right edge
            y_pos = 100  # 100px from top
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_pos},{y_pos}"
            
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Coffee Catcher")
            self._owns_screen = True

        # Asset directory
        self._assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        # Cup properties
        self.cup_width = 100
        self.cup_height = 66
        self.cup_speed = 8

        # Load cup images from assets
        self._cup_images = {
            0: self._load_image("0%.png", (self.cup_width, self.cup_height)),
            25: self._load_image("25%.png", (self.cup_width, self.cup_height)),
            50: self._load_image("50%.png", (self.cup_width, self.cup_height)),
            75: self._load_image("75%.png", (self.cup_width, self.cup_height)),
            100: self._load_image("100%.png", (self.cup_width, self.cup_height)),
        }

        # Load falling object images
        self._bean_image = self._load_image("bean.png", (30, 45))
        self._bomb_image = self._load_image("bomb.png", (40, 60))

        # Falling object sizes for collision detection
        self._bean_size = 40
        self._bomb_size = 40

        # Load background image
        self.background = self._load_image("game_background.png", (width, height))

        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)

        # Hand control
        self.hand_controller = None
        self.use_hand_control = use_hand_control
        if use_hand_control:
            try:
                from .hand_control import HandController
                self.hand_controller = HandController()
                if not self.hand_controller.start():
                    print("Warning: Could not start hand control, falling back to keyboard")
                    self.hand_controller = None
                    self.use_hand_control = False
            except ImportError:
                print("Warning: cvzone/opencv not installed, falling back to keyboard")
                print("Install with: pip install opencv-python cvzone")
                self.use_hand_control = False

        # Spawn event
        self.SPAWN_EVENT = pygame.USEREVENT + 1

        # Clock
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.reset()

    def _load_sound(self, filename: str) -> pygame.mixer.Sound:
        """Load a sound from the assets folder."""
        path = os.path.join(self._assets_dir, filename)
        try:
            sound = pygame.mixer.Sound(path)
            return sound
        except pygame.error as e:
            print(f"Warning: Could not load sound {filename}: {e}")
            return None


    def _load_image(self, filename: str, size: tuple) -> pygame.Surface:
        """Load an image from the assets folder and scale it."""
        path = os.path.join(self._assets_dir, filename)
        try:
            image = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(image, size)
        except pygame.error as e:
            print(f"Warning: Could not load {filename}: {e}")
            # Return a placeholder surface
            surface = pygame.Surface(size, pygame.SRCALPHA)
            surface.fill(self.BROWN)
            return surface

    def _get_cup_image(self) -> pygame.Surface:
        """Return the appropriate cup image based on current fullness."""
        if self.fullness >= 100:
            return self._cup_images[100]
        elif self.fullness >= 75:
            return self._cup_images[75]
        elif self.fullness >= 50:
            return self._cup_images[50]
        elif self.fullness >= 25:
            return self._cup_images[25]
        else:
            return self._cup_images[0]

    def reset(self) -> None:
        """Reset the game to initial state."""
        self.cup_x = self.width // 2 - self.cup_width // 2
        self.cup_y = self.height - 120
        self.fullness = 0
        self.game_over = False
        self.win = False
        self.objects = []
        self.start_time = time.time()
        self.remaining = self.game_time
        self.running = True

        # Reset spawn timer
        pygame.time.set_timer(self.SPAWN_EVENT, self.spawn_interval)

    def _spawn_object(self) -> None:
        """Spawn a new falling object (coffee or bomb)."""
        obj_type = "bomb" if random.random() < self.bomb_chance else "coffee"
        obj = {
            "type": obj_type,
            "x": random.randint(0, self.width - 40),
            "y": -40,
            "speed": random.randint(4, 7),
        }
        self.objects.append(obj)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle a single pygame event.

        Args:
            event: The pygame event to handle

        Returns:
            False if the game should quit, True otherwise
        """
        if event.type == pygame.QUIT:
            self.running = False
            return False

        if event.type == self.SPAWN_EVENT and not self.game_over:
            self._spawn_object()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return False

        return True

    def update(self) -> None:
        """Update game state for one frame."""
        if self.game_over:
            return

        # Update timer
        elapsed = time.time() - self.start_time
        self.remaining = max(0, self.game_time - elapsed)

        # Check if time ran out
        if self.remaining <= 0:
            self.game_over = True
            return

        # Move cup - hand control or keyboard
        if self.hand_controller is not None:
            direction = self.hand_controller.get_direction()
            if direction == 'left' and self.cup_x > 0:
                self.cup_x -= self.cup_speed
            elif direction == 'right' and self.cup_x < self.width - self.cup_width:
                self.cup_x += self.cup_speed
        else:
            # Keyboard control
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and self.cup_x > 0:
                self.cup_x -= self.cup_speed
            if keys[pygame.K_RIGHT] and self.cup_x < self.width - self.cup_width:
                self.cup_x += self.cup_speed

        # Update objects and check collisions
        cup_rect = pygame.Rect(self.cup_x, self.cup_y, self.cup_width, self.cup_height)

        for obj in self.objects[:]:
            obj["y"] += obj["speed"]
            obj_size = self._bean_size if obj["type"] == "coffee" else self._bomb_size
            obj_rect = pygame.Rect(obj["x"], obj["y"], obj_size, obj_size)

            # Check collision with cup
            if cup_rect.colliderect(obj_rect):
                if obj["type"] == "coffee":
                    self._load_sound("coffee_droplet_sound.MP3").play()
                    self.fullness += 10
                    if self.fullness >= 100:
                        self.fullness = 100
                        self.win = True
                        self.game_over = True
                else:  # bomb
                    self._load_sound("bomb_explode_sound.MP3").play()
                    self.game_over = True
                self.objects.remove(obj)
            # Remove if off screen
            elif obj["y"] > self.height:
                self.objects.remove(obj)

    def draw(self) -> None:
        """Draw the current game state to the screen."""
        # Draw background
        self.screen.blit(self.background, (0, 0))

        # Draw falling objects
        for obj in self.objects:
            if obj["type"] == "coffee":
                self.screen.blit(self._bean_image, (obj["x"], obj["y"]))
            else:  # bomb
                self.screen.blit(self._bomb_image, (obj["x"], obj["y"]))

        # Draw cup
        self.screen.blit(self._get_cup_image(), (self.cup_x, self.cup_y))

        # Draw UI
        text = self.font.render(f"Fullness: {self.fullness}%", True, self.BROWN)
        self.screen.blit(text, (10, 10))

        timer_color = self.RED if self.remaining <= 5 else self.BROWN
        timer_text = self.font.render(f"Time: {self.remaining:.1f}s", True, timer_color)
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 10, 10))

        # Draw game over or win screen
        if self.game_over:
            self._draw_end_screen()

    def _get_voucher_percentage(self) -> int:
        """Get voucher percentage based on cup fullness."""
        if self.fullness >= 90:
            return 20
        elif self.fullness >= 60:
            return 15
        elif self.fullness >= 30:
            return 10
        else:
            return 5

    def _draw_end_screen(self) -> None:
        """Draw the congratulations screen with voucher reward."""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        voucher = self._get_voucher_percentage()

        # Congratulations title
        congrats_text = self.big_font.render("Congratulations!", True, self.GREEN)
        self.screen.blit(
            congrats_text,
            (self.width // 2 - congrats_text.get_width() // 2, 80),
        )

        # Fullness result
        fullness_text = self.font.render(
            f"You filled {self.fullness}% of the cup!", True, self.WHITE
        )
        self.screen.blit(
            fullness_text,
            (self.width // 2 - fullness_text.get_width() // 2, 160),
        )

        # Voucher reward
        voucher_text = self.font.render(
            f"You won a {voucher}% discount voucher!", True, (255, 215, 0)  # Gold color
        )
        self.screen.blit(
            voucher_text,
            (self.width // 2 - voucher_text.get_width() // 2, 210),
        )

        # QR code placeholder
        qr_size = 250
        qr_x = self.width // 2 - qr_size // 2
        qr_y = 280
        
        # Try to load QR code image based on voucher percentage
        qr_filename = f"{voucher}%OFF.png"
        qr_path = os.path.join(self._assets_dir, qr_filename)
        
        if os.path.exists(qr_path):
            try:
                qr_image = pygame.image.load(qr_path).convert_alpha()
                qr_image = pygame.transform.scale(qr_image, (qr_size, qr_size))
                self.screen.blit(qr_image, (qr_x, qr_y))
            except pygame.error:
                self._draw_placeholder_qr(qr_x, qr_y, qr_size, voucher)
        else:
            self._draw_placeholder_qr(qr_x, qr_y, qr_size, voucher)

        # Scan instruction
        scan_text = self.font.render("Scan to claim your voucher!", True, self.WHITE)
        self.screen.blit(
            scan_text,
            (self.width // 2 - scan_text.get_width() // 2, qr_y + qr_size + 20),
        )

    def _draw_placeholder_qr(self, x: int, y: int, size: int, voucher: int) -> None:
        """Draw a placeholder QR code."""
        # White background
        pygame.draw.rect(self.screen, self.WHITE, (x, y, size, size))
        # Border
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, size, size), 3)
        # Placeholder text
        placeholder_font = pygame.font.Font(None, 24)
        text1 = placeholder_font.render("QR CODE", True, (100, 100, 100))
        text2 = placeholder_font.render(f"{voucher}% Voucher", True, (100, 100, 100))
        self.screen.blit(text1, (x + size // 2 - text1.get_width() // 2, y + size // 2 - 20))
        self.screen.blit(text2, (x + size // 2 - text2.get_width() // 2, y + size // 2 + 10))

    def _draw_instructions_screen(self) -> None:
        """Draw the hand control instructions screen."""
        # Background
        self.screen.fill(self.CREAM)
        
        # Title
        title_font = pygame.font.Font(None, 56)
        title = title_font.render("☕ Coffee Catcher ☕", True, self.BROWN)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 60))
        
        # Subtitle
        subtitle = self.font.render("Hand Control Instructions", True, self.DARK_BROWN)
        self.screen.blit(subtitle, (self.width // 2 - subtitle.get_width() // 2, 130))
        
        # Instructions box
        box_x, box_y = 50, 180
        box_width, box_height = self.width - 100, 380
        pygame.draw.rect(self.screen, self.WHITE, (box_x, box_y, box_width, box_height), border_radius=15)
        pygame.draw.rect(self.screen, self.BROWN, (box_x, box_y, box_width, box_height), 3, border_radius=15)
        
        # Instructions text
        instructions = [
            ("🖐️ Use ONE hand to control the cup", self.DARK_BROWN),
            ("", None),
            ("👈 MOVE LEFT:", self.BROWN),
            ("    Show ODD fingers (1, 3, 5)", (100, 100, 100)),
            ("", None),
            ("👉 MOVE RIGHT:", self.BROWN),
            ("    Show EVEN fingers (0, 2, 4)", (100, 100, 100)),
            ("", None),
            ("✊ START GAME:", self.GREEN),
            ("    Close your fist (0 fingers)", (100, 100, 100)),
        ]
        
        y_offset = box_y + 30
        instruction_font = pygame.font.Font(None, 32)
        for text, color in instructions:
            if text:
                rendered = instruction_font.render(text, True, color)
                self.screen.blit(rendered, (box_x + 30, y_offset))
            y_offset += 35
        
        # Bottom prompt
        prompt_font = pygame.font.Font(None, 40)
        prompt = prompt_font.render("✊ Close your fist to START! ✊", True, self.GREEN)
        self.screen.blit(prompt, (self.width // 2 - prompt.get_width() // 2, 600))
        
        # Keyboard alternative
        alt_text = self.font.render("(or press SPACE to start)", True, (150, 150, 150))
        self.screen.blit(alt_text, (self.width // 2 - alt_text.get_width() // 2, 650))

    def _show_instructions(self) -> bool:
        """
        Show instructions screen and wait for closed fist or SPACE to start.
        
        Returns:
            True if game should start, False if quit
        """
        waiting = True
        fist_hold_time = 0
        fist_hold_required = 0.5  # Hold fist for 0.5 seconds to start
        
        while waiting:
            self.clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_SPACE:
                        return True
            
            # Draw instructions
            self._draw_instructions_screen()
            
            # Check hand control for closed fist
            if self.hand_controller is not None:
                import cv2
                finger_count, frame = self.hand_controller.get_finger_count()
                
                if frame is not None:
                    # Show camera feed
                    self.hand_controller.show_frame(frame)
                
                if finger_count is not None and finger_count == 0:
                    fist_hold_time += 1/60  # Approximate frame time
                    
                    # Show progress
                    progress = min(fist_hold_time / fist_hold_required, 1.0)
                    bar_width = 200
                    bar_height = 20
                    bar_x = self.width // 2 - bar_width // 2
                    bar_y = 700
                    
                    # Draw progress bar background
                    pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), border_radius=10)
                    # Draw progress
                    pygame.draw.rect(self.screen, self.GREEN, (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=10)
                    # Draw border
                    pygame.draw.rect(self.screen, self.DARK_BROWN, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=10)
                    
                    if fist_hold_time >= fist_hold_required:
                        return True
                else:
                    fist_hold_time = 0
            
            pygame.display.flip()
        
        return True

    def run_frame(self) -> bool:
        """
        Run a single frame of the game (handle events, update, draw).

        Returns:
            False if the game should quit, True otherwise
        """
        self.clock.tick(60)

        # Handle events
        for event in pygame.event.get():
            if not self.handle_event(event):
                return False

        # Update
        self.update()

        # Draw
        self.draw()

        # Show hand control camera feed if enabled
        if self.hand_controller is not None:
            import cv2
            _, frame = self.hand_controller.get_finger_count()
            if frame is not None:
                if not self.hand_controller.show_frame(frame):
                    # User pressed 'q' in camera window
                    self.running = False
                    return False

        # Flip display
        pygame.display.flip()

        return self.running

    def run(self) -> dict:
        """
        Run the game until completion or quit.

        Returns:
            Dictionary with game results: {'win': bool, 'fullness': int}
        """
        # Show instructions screen if hand control is enabled
        if self.use_hand_control and self.hand_controller is not None:
            if not self._show_instructions():
                # User quit during instructions
                return {"win": False, "fullness": 0}
        
        self.reset()

        while self.running:
            if not self.run_frame():
                break

        # Stop spawn timer
        pygame.time.set_timer(self.SPAWN_EVENT, 0)

        return {"win": self.win, "fullness": self.fullness}

    def quit(self) -> None:
        """Clean up resources."""
        pygame.time.set_timer(self.SPAWN_EVENT, 0)
        if self.hand_controller is not None:
            self.hand_controller.stop()
        if self._owns_screen:
            pygame.quit()


def main():
    """Entry point for running the game standalone."""
    game = CoffeeGame()
    result = game.run()
    print(f"Game ended - Win: {result['win']}, Fullness: {result['fullness']}%")
    game.quit()


if __name__ == "__main__":
    main()
