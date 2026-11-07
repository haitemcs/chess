import sys, os, time, shutil
from pathlib import Path

import cv2
import chess
import chess.engine
import numpy as np
import mss

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QFont, QPolygonF
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QListWidget,
    QGroupBox, QFileDialog, QSpinBox, QMessageBox
)

from chessbot.vision import board_detect
from chessbot.vision.recognizer import Recognizer
from chessbot.vision.position import (
    infer_white_at_bottom, grid_to_board, board_to_grid
)

ROOT = Path(__file__).parent
REPO = ROOT / "OpenCV-Chess-Bot"
MODEL = REPO / "models" / "piece_classifier.onnx"

def find_stockfish():
    candidates = [
        shutil.which("stockfish"),
        r"C:\Program Files\Stockfish\stockfish.exe",
        r"C:\Program Files (x86)\Stockfish\stockfish.exe",
        str(Path.home() / "AppData/Local/Programs/Stockfish/stockfish.exe"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return str(path)
    return None

def engine_preference(scores):
    # Softmax of evaluation differences: relative engine preference,
    # not a statistically calibrated probability of a move being played.
    values = np.array(scores, dtype=float)
    values = np.clip((values - values.max()) / 180.0, -30, 0)
    weights = np.exp(values)
    return weights / weights.sum() * 100

class BoardOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.rect_board = None
        self.white_bottom = True
        self.moves = []
        self.showFullScreen()

    def update_board(self, rect, white_bottom, moves):
        self.rect_board = rect
        self.white_bottom = white_bottom
        self.moves = moves
        self.update()

    def center(self, square):
        if self.rect_board is None:
            return QPointF()

        x, y, w, h = self.rect_board
        f = chess.square_file(square)
        r = chess.square_rank(square)

        if self.white_bottom:
            col, row = f, 7-r
        else:
            col, row = 7-f, r

        return QPointF(
            x + (col + 0.5) * w / 8,
            y + (row + 0.5) * h / 8
        )

    def paintEvent(self, event):
        if not self.rect_board:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        colors = [
            QColor("#16c784"),
            QColor("#f5c542"),
            QColor("#f28c28"),
            QColor("#df5b5b"),
            QColor("#a78bfa")
        ]

        for i, (move, probability, san) in enumerate(self.moves):
            color = colors[min(i, len(colors)-1)]
            start = self.center(move.from_square)
            end = self.center(move.to_square)

            width = max(3, 9 - i)

            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawLine(start, end)

            # Arrow head
            direction = end - start
            length = max(1.0, (direction.x()**2 + direction.y()**2)**0.5)
            ux, uy = direction.x()/length, direction.y()/length
            px, py = -uy, ux
            head = 17 - i
            points = QPolygonF([
                end,
                QPointF(end.x()-head*ux+head*.55*px,
                        end.y()-head*uy+head*.55*py),
                QPointF(end.x()-head*ux-head*.55*px,
                        end.y()-head*uy-head*.55*py)
            ])
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(points)

            # Percentage label close to destination square
            label_pos = QPointF(end.x()+9, end.y()-12)
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#111111"), 4))
            painter.drawText(label_pos, f"{probability:.0f}%")
            painter.setPen(QPen(color, 1))
            painter.drawText(label_pos, f"{probability:.0f}%")

        painter.end()

class Analyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Analyzer")
        self.resize(430, 650)

        self.overlay = BoardOverlay()
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

        self.recognizer = Recognizer(str(MODEL))
        self.board = None
        self.last_grid = None
        self.last_fen = None
        self.board_region = None
        self.white_bottom = True
        self.engine = None
        self.engine_path = find_stockfish()
        self.running = True

        self.build_ui()

        if not self.engine_path:
            self.choose_engine()

        if self.engine_path:
            self.start_engine()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scan_screen)
        self.timer.start(1100)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("CHESS ANALYZER")
        title.setStyleSheet("font-size:25px;font-weight:900;color:#16c784")
        layout.addWidget(title)

        self.status = QLabel("Searching for a chessboard...")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#aaa;padding:8px")
        layout.addWidget(self.status)

        controls = QHBoxLayout()

        self.turn = QComboBox()
        self.turn.addItem("White to move", chess.WHITE)
        self.turn.addItem("Black to move", chess.BLACK)

        controls.addWidget(QLabel("Initial turn:"))
        controls.addWidget(self.turn)
        layout.addLayout(controls)

        self.depth = QSpinBox()
        self.depth.setRange(8, 24)
        self.depth.setValue(13)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Engine depth"))
        depth_row.addWidget(self.depth)
        layout.addLayout(depth_row)

        self.evaluation = QLabel("Evaluation: --")
        self.evaluation.setStyleSheet("font-size:23px;font-weight:bold")
        layout.addWidget(self.evaluation)

        self.best = QLabel("Best move: --")
        self.best.setStyleSheet("font-size:19px;color:#16c784;font-weight:bold")
        layout.addWidget(self.best)

        group = QGroupBox("ENGINE PREFERENCE")
        group_layout = QVBoxLayout(group)
        self.candidates = QListWidget()
        self.candidates.setStyleSheet("font-size:15px;padding:8px")
        group_layout.addWidget(self.candidates)
        layout.addWidget(group)

        self.fen = QLabel("Position: waiting...")
        self.fen.setWordWrap(True)
        self.fen.setStyleSheet("color:#999;font-size:11px")
        layout.addWidget(self.fen)

        buttons = QHBoxLayout()

        self.scan_button = QPushButton("Rescan board")
        self.scan_button.clicked.connect(self.force_scan)
        buttons.addWidget(self.scan_button)

        self.engine_button = QPushButton("Select Stockfish")
        self.engine_button.clicked.connect(self.choose_engine)
        buttons.addWidget(self.engine_button)

        layout.addLayout(buttons)

        self.setStyleSheet("""
            QMainWindow,QWidget {
                background:#17191f;color:#eee;
                font-family:Arial;
            }
            QGroupBox {
                border:1px solid #444;border-radius:8px;
                margin-top:12px;padding:12px;font-weight:bold;
            }
            QPushButton {
                background:#292d36;border:1px solid #555;
                padding:10px;border-radius:6px;
            }
            QPushButton:hover {background:#3b414c;}
            QListWidget {
                background:#20232b;border:0;border-radius:6px;
            }
            QComboBox,QSpinBox {
                background:#292d36;padding:7px;border:1px solid #555;
            }
        """)

    def choose_engine(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stockfish executable", "",
            "Executable (*.exe)"
        )
        if path:
            self.engine_path = path
            self.start_engine()

    def start_engine(self):
        try:
            if self.engine:
                self.engine.quit()
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.engine_path
            )
            self.engine.configure({
                "Threads": 2,
                "Hash": 256
            })
            self.status.setText("Stockfish connected. Searching for board...")
        except Exception as exc:
            self.engine = None
            self.status.setText(f"Engine error: {exc}")

    def force_scan(self):
        self.last_grid = None
        self.last_fen = None
        self.board = None
        self.scan_screen()

    def scan_screen(self):
        if not self.running:
            return

        # Hide our overlay before taking the screenshot.
        self.overlay.hide()
        QApplication.processEvents()

        try:
            shot = np.array(self.sct.grab(self.monitor))
            image = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

            region = board_detect.locate_board(image)

            if region is None:
                self.status.setText(
                    "No board detected. Make sure the entire 2D board is visible."
                )
                return

            x0, y0, x1, y1 = region
            crop = image[y0:y1, x0:x1]

            if crop.size == 0:
                return

            grid, confidence = self.recognizer.classify_squares(crop)
            min_conf = float(confidence.min())
            self.white_bottom = infer_white_at_bottom(grid)

            if min_conf < 0.75:
                self.status.setText(
                    f"Recognition uncertain ({min_conf:.0%}). "
                    "Trying again; no new move will be recommended."
                )
                return

            if self.board is None:
                self.board = grid_to_board(
                    grid, self.white_bottom, self.turn.currentData()
                )
                self.last_grid = grid
                self.analyze_position()

            elif grid != self.last_grid:
                self.update_position(grid)

            self.board_region = (
                x0 + int(self.monitor["left"]),
                y0 + int(self.monitor["top"]),
                x1-x0,
                y1-y0
            )

            self.status.setText(
                f"Board detected | Recognition confidence: {min_conf:.0%}"
            )

            if self.board:
                self.overlay.update_board(
                    self.board_region,
                    self.white_bottom,
                    getattr(self, "best_moves", [])
                )

        except Exception as exc:
            self.status.setText(f"Recognition error: {exc}")

        finally:
            self.overlay.show()
            self.overlay.raise_()

    def update_position(self, grid):
        # First try to identify the change as one legal chess move.
        matches = []

        for move in list(self.board.legal_moves):
            test = self.board.copy()
            test.push(move)

            if board_to_grid(test, self.white_bottom) == grid:
                matches.append(move)

        if len(matches) == 1:
            self.board.push(matches[0])
        else:
            # Resynchronize if the game was already in progress or
            # several frames were missed.
            next_turn = not self.board.turn
            try:
                self.board = grid_to_board(
                    grid, self.white_bottom, next_turn
                )
            except ValueError:
                self.status.setText(
                    "Position could not be validated. Waiting for a clearer frame."
                )
                return

        self.last_grid = grid
        self.analyze_position()

    def analyze_position(self):
        if not self.engine or not self.board:
            self.best_moves = []
            return

        if self.board.is_game_over():
            self.best.setText("Game finished")
            self.evaluation.setText(
                "Result: " + self.board.result()
            )
            self.best_moves = []
            return

        try:
            depth = self.depth.value()
            results = self.engine.analyse(
                self.board,
                chess.engine.Limit(depth=depth),
                multipv=5
            )

            if isinstance(results, dict):
                results = [results]

            candidates = []
            scores = []

            for info in results:
                if not info.get("pv"):
                    continue

                move = info["pv"][0]
                score = info["score"].pov(self.board.turn).score(
                    mate_score=100000
                )
                scores.append(score)
                candidates.append((move, self.board.san(move)))

            if not candidates:
                return

            probabilities = engine_preference(scores)
            self.best_moves = []

            self.candidates.clear()

            for (move, san), probability, score in zip(
                candidates, probabilities, scores
            ):
                self.best_moves.append((move, probability, san))

                self.candidates.addItem(
                    f"{san:<9} {probability:5.1f}%     "
                    f"{score/100:+.2f}"
                )

            best_score = scores[0]
            self.best.setText(
                f"BEST MOVE: {candidates[0][1]} "
                f"({probabilities[0]:.1f}%)"
            )
            self.evaluation.setText(
                f"Evaluation: {best_score/100:+.2f}"
            )
            self.fen.setText("FEN: " + self.board.fen())

            self.last_fen = self.board.fen()

        except Exception as exc:
            self.status.setText(f"Engine analysis error: {exc}")

    def closeEvent(self, event):
        self.running = False
        self.timer.stop()
        self.overlay.close()

        try:
            if self.engine:
                self.engine.quit()
            self.sct.close()
        except Exception:
            pass

        event.accept()

def main():
    app = QApplication(sys.argv)
    window = Analyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
