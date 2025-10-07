use macroquad::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

fn window_conf() -> Conf {
    Conf {
        window_title: "Pong".to_owned(),
        fullscreen: false,
        ..Default::default()
    }
}

fn draw_player(player: &Player) {
    draw_line(
        player.pos.x - player.height * player.angle.cos() / 2.0,
        player.pos.y - player.height * player.angle.sin() / 2.0,
        player.pos.x + player.height * player.angle.cos() / 2.0,
        player.pos.y + player.height * player.angle.sin() / 2.0,
        player.width,
        WHITE,
    );
}

fn draw_ball(ball: &Ball) {
    draw_circle(ball.pos.x, ball.pos.y, ball.radius, WHITE);
}

struct Ball {
    pos: Vec2,
    speed: Vec2,
    radius: f32,
    mass: f32,
}

struct Player {
    pos: Vec2,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    width: f32,
    height: f32,
    speed: Vec2,
    linear_acc: f32,
    linear_friction: f32,
    angle: f32,
    angular_velocity: f32,
    angular_acc: f32,
    angular_friction: f32,
    mass: f32,
}

impl Ball {
    fn update(&mut self, delta_time: f32) -> i32 {
        let mut out = 0;
        if self.pos.x - self.radius < 0.0 {
            out = 1;
            self.speed.x = self.speed.x.abs();
        }
        if self.pos.x + self.radius > screen_width() {
            out = 2;
            self.speed.x = -self.speed.x.abs();
        }
        if self.pos.y - self.radius < 0.0 {
            self.speed.y = self.speed.y.abs();
        }
        if self.pos.y + self.radius > screen_height() {
            self.speed.y = -self.speed.y.abs();
        }
        self.pos.x += self.speed.x * delta_time;
        self.pos.y += self.speed.y * delta_time;

        out
    }

    fn check_collision(&mut self, player: &mut Player) -> bool {
        let dc = self.pos - player.pos;
        let r = player.height * vec2(player.angle.cos(), player.angle.sin()) / 2.0;
        let d1s = dc + r;
        let d1 = d1s.dot(d1s).sqrt();

        let d2s = dc - r;
        let d2 = d2s.dot(d2s).sqrt();

        let buffer = player.width / 2.0 + self.radius / 2.0;
        let mut out = false;
        if (d1 + d2 > player.height - buffer) & (d1 + d2 < player.height + buffer) {
            let n = vec2(-player.angle.sin(), player.angle.cos());

            let d = d1 * player.height / (d1 + d2) - player.height / 2.0;
            let v_c = player.speed + player.angular_velocity * d * n;

            let v_rel = self.speed - v_c;

            let i = player.mass * player.height * player.height / 12.0;
            let j = -2.0 * v_rel.dot(n) / (1.0 / self.mass + 1.0 / player.mass + (d * d) / i);

            self.speed += j * n / self.mass;

            player.speed -= j * n / player.mass;
            player.angular_velocity -= j * d / i;
            println!("Collision detected at time {}", get_time());
            out = true;
        }
        out
    }
}

impl Player {
    fn automatic_movement(&mut self, ball: &Ball, delta_time: f32) {
        if ball.pos.x > self.pos.x {
            if ball.pos.y > self.pos.y {
                self.speed.y += self.linear_acc * delta_time;
            } else if ball.pos.y < self.pos.y {
                self.speed.y -= self.linear_acc * delta_time;
            }
        } else if (ball.pos.y > self.pos.y) & (ball.pos.y < self.pos.y + self.height / 2.0) {
            self.speed.y -= self.linear_acc * delta_time;
        } else if (ball.pos.y < self.pos.y) & (ball.pos.y > self.pos.y - self.height / 2.0) {
            self.speed.y += self.linear_acc * delta_time;
        }
    }

    fn manual_movement(&mut self, delta_time: f32) {
        if is_key_down(KeyCode::W) {
            self.speed.y -= self.linear_acc * delta_time;
        }
        if is_key_down(KeyCode::S) {
            self.speed.y += self.linear_acc * delta_time;
        }
        if is_key_down(KeyCode::A) {
            self.speed.x -= self.linear_acc * delta_time;
        }
        if is_key_down(KeyCode::D) {
            self.speed.x += self.linear_acc * delta_time;
        }
        if is_key_down(KeyCode::Q) {
            self.angular_velocity -= self.angular_acc * delta_time;
        }
        if is_key_down(KeyCode::E) {
            self.angular_velocity += self.angular_acc * delta_time;
        }
    }

    fn update(&mut self, delta_time: f32) {
        self.pos.x += self.speed.x * delta_time;
        self.pos.y += self.speed.y * delta_time;
        self.angle += self.angular_velocity * delta_time;
        if self.angle > 2.0 * PI {
            self.angle -= 2.0 * PI;
        }
        if self.angle < 0.0 {
            self.angle += 2.0 * PI;
        }

        if self.pos.x < self.min_x {
            self.pos.x = self.min_x;
            self.speed.x = -self.speed.x;
        } else if self.pos.x > self.max_x {
            self.pos.x = self.max_x;
            self.speed.x = -self.speed.x;
        } else if self.pos.y < self.min_y {
            self.pos.y = self.min_y;
            self.speed.y = -self.speed.y;
        } else if self.pos.y > self.max_y {
            self.pos.y = self.max_y;
            self.speed.y = -self.speed.y;
        }

        self.angular_velocity -= self.angular_friction * self.angular_velocity * delta_time;
        self.speed -= self.linear_friction * self.speed * delta_time;
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut ball = Ball {
        pos: vec2(screen_width() / 2.0, screen_height() / 2.0),
        speed: vec2(100.0, 19.0),
        radius: 10.0,
        mass: 1.0,
    };
    let mut player1 = Player {
        pos: vec2(9.0 * screen_width() / 10.0, screen_height() / 2.0),
        max_x: screen_width(),
        min_x: 3.0 * screen_width() / 5.0,
        max_y: screen_height(),
        min_y: 0.0,
        width: 5.0,
        height: 150.0,
        speed: vec2(0.0, 0.0),
        linear_acc: 500.0,
        linear_friction: 0.1,
        angle: PI / 2.0,
        angular_velocity: 0.0,
        angular_acc: 5.0,
        angular_friction: 0.1,
        mass: 10.0,
    };
    let mut player2 = Player {
        pos: vec2(screen_width() / 10.0, screen_height() / 2.0),
        max_x: 2.0 * screen_width() / 5.0,
        min_x: 0.0,
        max_y: screen_height(),
        min_y: 0.0,
        width: 5.0,
        height: 150.0,
        speed: vec2(0.0, 0.0),
        linear_acc: 500.0,
        linear_friction: 0.1,
        angle: PI / 2.0,
        angular_velocity: 0.0,
        angular_acc: 5.0,
        angular_friction: 0.1,
        mass: 10.0,
    };

    let mut points_player1 = 0;
    let mut points_player2 = 0;
    let mut wall_hit;
    let mut delta_time;
    let mut hit_time = Instant::now();
    let mut point_time = Instant::now();
    loop {
        clear_background(BLACK);

        delta_time = get_frame_time();

        player2.automatic_movement(&ball, delta_time);
        player1.manual_movement(delta_time);

        if hit_time.elapsed().as_millis() > 50 {
            if ball.check_collision(&mut player1) | ball.check_collision(&mut player2) {
                hit_time = Instant::now();
            }
        }

        wall_hit = ball.update(delta_time);
        if point_time.elapsed().as_millis() > 100 {
            if wall_hit == 1 {
                points_player1 += 1;
                point_time = Instant::now();
            } else if wall_hit == 2 {
                points_player2 += 1;
                point_time = Instant::now();
            }
        }

        player1.update(delta_time);
        player2.update(delta_time);

        draw_player(&player1);
        draw_player(&player2);
        draw_ball(&ball);

        draw_text(
            &format!("Player 1: {}", points_player1),
            screen_width() - 100.0,
            10.0,
            20.0,
            WHITE,
        );
        draw_text(
            &format!("Player 2: {}", points_player2),
            10.0,
            10.0,
            20.0,
            WHITE,
        );

        if points_player1 >= 10 || points_player2 >= 10 {
            if points_player1 > points_player2 {
                draw_text(
                    &format!("Player 1 wins!"),
                    screen_width() / 2.0 - 100.0,
                    screen_height() / 2.0,
                    30.0,
                    WHITE,
                );
            } else {
                draw_text(
                    &format!("Player 2 wins!"),
                    screen_width() / 2.0 - 100.0,
                    screen_height() / 2.0,
                    30.0,
                    WHITE,
                );
            }
        }

        next_frame().await
    }
}
