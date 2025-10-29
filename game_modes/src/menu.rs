use macroquad::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum GameStateType {
    Welcome,
    Playing,
    GameOver { winner: String },
}

pub struct MenuSystem {
    state: GameStateType,
    buttons: Vec<Button>,
}

#[derive(Debug, Clone)]
pub struct Button {
    pub rect: Rect,
    pub text: String,
    pub action: ButtonAction,
    pub is_hovered: bool,
    pub is_pressed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ButtonAction {
    StartGame,
    Quit,
    NewGame,
}

impl MenuSystem {
    pub fn new() -> Self {
        Self {
            state: GameStateType::Welcome,
            buttons: Vec::new(),
        }
    }

    pub fn get_state(&self) -> &GameStateType {
        &self.state
    }

    pub fn set_state(&mut self, state: GameStateType) {
        self.state = state;
        self.update_buttons();
    }

    pub fn update(&mut self) -> Option<ButtonAction> {
        self.update_buttons();
        self.handle_input()
    }

    fn update_buttons(&mut self) {
        self.buttons.clear();

        match &self.state {
            GameStateType::Welcome => {
                self.buttons = vec![
                    Button::new(
                        self.get_centered_button_rect(0, 2),
                        "START GAME".to_string(),
                        ButtonAction::StartGame,
                    ),
                    Button::new(
                        self.get_centered_button_rect(1, 2),
                        "QUIT".to_string(),
                        ButtonAction::Quit,
                    ),
                ];
            }
            GameStateType::GameOver { .. } => {
                self.buttons = vec![
                    Button::new(
                        self.get_centered_button_rect(0, 2),
                        "NEW GAME".to_string(),
                        ButtonAction::NewGame,
                    ),
                    Button::new(
                        self.get_centered_button_rect(1, 2),
                        "QUIT".to_string(),
                        ButtonAction::Quit,
                    ),
                ];
            }
            GameStateType::Playing => {}
        }
    }

    fn get_centered_button_rect(&self, index: i32, total: i32) -> Rect {
        let button_width = 200.0;
        let button_height = 60.0;
        let spacing = 20.0;

        let total_height = (total as f32) * button_height + ((total - 1) as f32) * spacing;
        let start_y = (screen_height() - total_height) / 2.0;

        let x = (screen_width() - button_width) / 2.0;
        let y = start_y + (index as f32) * (button_height + spacing);

        Rect::new(x, y, button_width, button_height)
    }

    fn handle_input(&mut self) -> Option<ButtonAction> {
        let mouse_pos = mouse_position();
        let mouse_clicked = is_mouse_button_pressed(MouseButton::Left);

        let touches = touches();
        let touch_pressed =
            !touches.is_empty() && touches.iter().any(|t| t.phase == TouchPhase::Started);
        let touch_pos = if !touches.is_empty() {
            Some((touches[0].position.x, touches[0].position.y))
        } else {
            None
        };

        for button in &mut self.buttons {
            let pos = if let Some(touch_pos) = touch_pos {
                touch_pos
            } else {
                mouse_pos
            };

            button.is_hovered = button.rect.contains(Vec2::new(pos.0, pos.1));

            if button.is_hovered && (mouse_clicked || touch_pressed) {
                button.is_pressed = true;
                return Some(button.action.clone());
            }
        }

        if matches!(
            self.state,
            GameStateType::Welcome | GameStateType::GameOver { .. }
        ) {
            if is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::Space) {
                if let Some(button) = self.buttons.first() {
                    return Some(button.action.clone());
                }
            }
            if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
                return Some(ButtonAction::Quit);
            }
        }

        None
    }

    pub fn render(&self) {
        match &self.state {
            GameStateType::Welcome => self.render_welcome_screen(),
            GameStateType::GameOver { winner } => self.render_game_over_screen(winner),
            GameStateType::Playing => {}
        }
    }

    fn render_welcome_screen(&self) {
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            screen_height(),
            Color::new(0.0, 0.0, 0.0, 0.7),
        );

        // Title
        let title_font_size = 80.0;
        let title = "PFONG";
        let title_dims = measure_text(title, None, title_font_size as u16, 1.0);
        let title_x = (screen_width() - title_dims.width) / 2.0;
        let title_y = screen_height() / 3.0;

        draw_text(title, title_x, title_y, title_font_size, WHITE);

        self.render_buttons();

        let instructions = "Use ENTER or SPACE to start, ESC to quit";
        let inst_font_size = 16.0;
        let inst_dims = measure_text(instructions, None, inst_font_size as u16, 1.0);
        let inst_x = (screen_width() - inst_dims.width) / 2.0;
        let inst_y = screen_height() - 25.0;

        draw_text(instructions, inst_x, inst_y, inst_font_size, GRAY);
    }

    fn render_game_over_screen(&self, winner: &str) {
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            screen_height(),
            Color::new(0.0, 0.0, 0.0, 0.7),
        );

        let title_font_size = 60.0;
        let title = "GAME OVER";
        let title_dims = measure_text(title, None, title_font_size as u16, 1.0);
        let title_x = (screen_width() - title_dims.width) / 2.0;
        let title_y = screen_height() / 3.0;

        draw_text(title, title_x, title_y, title_font_size, WHITE);

        let winner_font_size = 32.0;
        let winner_text = format!("{} Wins!", winner);
        let winner_dims = measure_text(&winner_text, None, winner_font_size as u16, 1.0);
        let winner_x = (screen_width() - winner_dims.width) / 2.0;
        let winner_y = title_y + 80.0;

        let winner_color = if winner.contains("Player 1") {
            Color::new(0.3, 0.3, 1.0, 1.0)
        } else {
            Color::new(1.0, 0.3, 0.3, 1.0)
        };

        draw_text(
            &winner_text,
            winner_x,
            winner_y,
            winner_font_size,
            winner_color,
        );

        self.render_buttons();

        let instructions = "Use ENTER for new game, ESC to quit";
        let inst_font_size = 16.0;
        let inst_dims = measure_text(instructions, None, inst_font_size as u16, 1.0);
        let inst_x = (screen_width() - inst_dims.width) / 2.0;
        let inst_y = screen_height() - 25.0;

        draw_text(instructions, inst_x, inst_y, inst_font_size, GRAY);
    }

    fn render_buttons(&self) {
        let font_size = 24.0;

        for button in &self.buttons {
            let bg_color = if button.is_hovered {
                Color::new(0.3, 0.3, 0.3, 0.8)
            } else {
                Color::new(0.1, 0.1, 0.1, 0.8)
            };

            draw_rectangle(
                button.rect.x,
                button.rect.y,
                button.rect.w,
                button.rect.h,
                bg_color,
            );

            let border_color = if button.is_hovered { WHITE } else { LIGHTGRAY };

            draw_rectangle_lines(
                button.rect.x,
                button.rect.y,
                button.rect.w,
                button.rect.h,
                2.0,
                border_color,
            );

            let text_dims = measure_text(&button.text, None, font_size as u16, 1.0);
            let text_x = button.rect.x + (button.rect.w - text_dims.width) / 2.0;
            let text_y = button.rect.y + (button.rect.h + text_dims.height) / 2.0;

            let text_color = if button.is_hovered { WHITE } else { LIGHTGRAY };

            draw_text(&button.text, text_x, text_y, font_size, text_color);
        }
    }
}

impl Button {
    pub fn new(rect: Rect, text: String, action: ButtonAction) -> Self {
        Self {
            rect,
            text,
            action,
            is_hovered: false,
            is_pressed: false,
        }
    }
}

impl Default for MenuSystem {
    fn default() -> Self {
        Self::new()
    }
}
