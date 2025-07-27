"""
Main Window Module
=================

Main application window with CustomTkinter UI components.
"""

import customtkinter as ctk
import logging
from typing import Optional, Callable
from PIL import Image, ImageTk
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MainWindow:
    """Main application window with modern UI"""
    
    def __init__(self, title: str = "SafeStep - Indoor Navigation System", 
                 width: int = 1400, height: int = 900):
        self.root = ctk.CTk()
        self.title = title
        self.width = width
        self.height = height
        
        # UI components
        self.video_label = None
        self.instruction_label = None
        self.fps_label = None
        self.detection_count_label = None
        self.system_status_label = None
        self.stats_text = None
        self.history_text = None
        self.confidence_slider = None
        self.confidence_label = None
        self.voice_enabled = None
        self.mode_var = None
        
        # Video display
        self.video_image = None
        self.video_photo = None
        
        # Callbacks
        self.on_mode_change_callback = None
        self.on_confidence_change_callback = None
        self.on_voice_toggle_callback = None
        self.on_emergency_callback = None
        self.on_export_callback = None
        self.on_clear_callback = None
        self.on_closing_callback = None
        
        self._init_window()
        self._create_ui()
    
    def _init_window(self):
        """Initialize main window"""
        self.root.title(self.title)
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Configure CustomTkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind('<Escape>', lambda e: self._toggle_fullscreen())
        self.root.bind('<F11>', lambda e: self._toggle_fullscreen())
    
    def _create_ui(self):
        """Create UI layout"""
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Create UI sections
        self._create_navigation_bar()
        self._create_content_area()
        self._create_status_bar()
    
    def _create_navigation_bar(self):
        """Create top navigation bar"""
        nav_frame = ctk.CTkFrame(self.main_frame, height=70, corner_radius=10)
        nav_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        nav_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        title_frame.pack(side="left", padx=20)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🧭 SAFESTEP",
            font=ctk.CTkFont(size=28, weight="bold", family="Helvetica"),
            text_color="#4cc9f0"
        )
        title_label.pack(side="top")
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Advanced Indoor Navigation System",
            font=ctk.CTkFont(size=14, family="Helvetica"),
            text_color="#a0a0a0"
        )
        subtitle_label.pack(side="top", pady=(0, 5))
        
        # Right controls
        controls_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        controls_frame.pack(side="right", padx=20)
        
        # Navigation mode selector
        mode_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        mode_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(mode_frame, text="Mode:").pack(side="left", padx=(0, 5))
        self.mode_var = ctk.StringVar(value="Autonomous")
        mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            variable=self.mode_var,
            values=["Autonomous", "Guided", "Exploration", "Emergency"],
            command=self._on_mode_change,
            button_color="#4361ee",
            fg_color="#3a0ca3",
            dropdown_fg_color="#2b2d42"
        )
        mode_menu.pack(side="left")
        
        # Emergency button
        emergency_btn = ctk.CTkButton(
            controls_frame,
            text="🚨 EMERGENCY",
            command=self._on_emergency,
            fg_color="#d90429",
            hover_color="#ef233c",
            width=120,
            corner_radius=8,
            font=ctk.CTkFont(weight="bold")
        )
        emergency_btn.pack(side="left", padx=(10, 0))
    
    def _create_content_area(self):
        """Create main content area"""
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Create panels
        self._create_control_panel()
        self._create_video_panel()
        self._create_analytics_panel()
    
    def _create_control_panel(self):
        """Create left control panel"""
        control_frame = ctk.CTkFrame(self.content_frame, width=280, corner_radius=10)
        control_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        control_frame.grid_propagate(False)
        control_frame.grid_rowconfigure(4, weight=1)
        
        # Panel title
        panel_title = ctk.CTkLabel(
            control_frame, 
            text="CONTROL PANEL",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        panel_title.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        
        # Detection Settings
        detect_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        detect_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            detect_frame, 
            text="Detection Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Confidence threshold slider
        slider_frame = ctk.CTkFrame(detect_frame, fg_color="transparent")
        slider_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(slider_frame, text="Confidence:").pack(side="left")
        self.confidence_label = ctk.CTkLabel(slider_frame, text="0.5")
        self.confidence_label.pack(side="right")
        
        self.confidence_slider = ctk.CTkSlider(
            detect_frame, 
            from_=0.1, 
            to=1.0, 
            number_of_steps=18,
            command=self._on_confidence_change,
            button_color="#4361ee",
            button_hover_color="#3a0ca3"
        )
        self.confidence_slider.set(0.5)
        self.confidence_slider.pack(fill="x", padx=10, pady=(0, 10))
        
        # Voice Controls
        voice_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        voice_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            voice_frame, 
            text="Voice Assistant",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        self.voice_enabled = ctk.CTkSwitch(
            voice_frame, 
            text="Enable Voice Commands",
            command=self._on_voice_toggle,
            switch_width=50,
            switch_height=25,
            progress_color="#4361ee"
        )
        self.voice_enabled.pack(padx=10, pady=(5, 10))
        self.voice_enabled.select()
        
        # Quick Actions
        actions_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        actions_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            actions_frame, 
            text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        actions = [
            ("🚪 Find Exit", lambda: self._on_action("find_exit")),
            ("🚻 Find Restroom", lambda: self._on_action("find_restroom")),
            ("🛗 Find Elevator", lambda: self._on_action("find_elevator")),
            ("📍 Current Location", lambda: self._on_action("announce_location")),
            ("🔄 Recalibrate", lambda: self._on_action("recalibrate"))
        ]
        
        for text, command in actions:
            btn = ctk.CTkButton(
                actions_frame, 
                text=text, 
                command=command,
                corner_radius=6,
                fg_color="#2b2d42",
                hover_color="#4a4e69",
                anchor="w"
            )
            btn.pack(fill="x", padx=10, pady=5)
        
        # System Status
        status_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        status_frame.grid(row=4, column=0, padx=10, pady=10, sticky="sew")
        
        ctk.CTkLabel(
            status_frame, 
            text="System Status",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        status_text = ctk.CTkTextbox(status_frame, height=80, activate_scrollbars=False)
        status_text.pack(fill="x", padx=10, pady=(0, 10))
        status_text.insert("1.0", "System operational\nAll sensors active")
        status_text.configure(state="disabled")
    
    def _create_video_panel(self):
        """Create center video panel"""
        video_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        video_frame.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(0, weight=1)
        
        # Video display container
        video_container = ctk.CTkFrame(video_frame)
        video_container.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Video display
        self.video_label = ctk.CTkLabel(video_container, text="", corner_radius=8)
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Overlay information
        overlay_frame = ctk.CTkFrame(video_frame, height=120, corner_radius=8)
        overlay_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Current instruction
        self.instruction_label = ctk.CTkLabel(
            overlay_frame,
            text="System Ready - Voice commands active",
            font=ctk.CTkFont(size=18, weight="bold"),
            wraplength=600
        )
        self.instruction_label.pack(pady=10, padx=20, anchor="w")
        
        # Navigation info
        nav_info_frame = ctk.CTkFrame(overlay_frame, fg_color="transparent")
        nav_info_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.distance_label = ctk.CTkLabel(
            nav_info_frame, 
            text="Distance: --",
            font=ctk.CTkFont(size=14)
        )
        self.distance_label.pack(side="left", padx=10)
        
        self.direction_label = ctk.CTkLabel(
            nav_info_frame, 
            text="Direction: --",
            font=ctk.CTkFont(size=14)
        )
        self.direction_label.pack(side="left", padx=10)
        
        self.safety_label = ctk.CTkLabel(
            nav_info_frame, 
            text="✅ Safe",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.safety_label.pack(side="right", padx=10)
    
    def _create_analytics_panel(self):
        """Create right analytics panel"""
        analytics_frame = ctk.CTkFrame(self.content_frame, width=350, corner_radius=10)
        analytics_frame.grid(row=0, column=2, padx=(10, 0), pady=0, sticky="nsew")
        analytics_frame.grid_propagate(False)
        analytics_frame.grid_rowconfigure(1, weight=1)
        
        # Panel title
        panel_title = ctk.CTkLabel(
            analytics_frame, 
            text="ANALYTICS DASHBOARD",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        panel_title.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        
        # Tab view for different analytics
        self.analytics_tabs = ctk.CTkTabview(analytics_frame)
        self.analytics_tabs.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        # Real-time stats tab
        realtime_tab = self.analytics_tabs.add("Realtime")
        realtime_tab.grid_columnconfigure(0, weight=1)
        
        stats_frame = ctk.CTkScrollableFrame(realtime_tab)
        stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.stats_text = ctk.CTkTextbox(stats_frame, height=300)
        self.stats_text.pack(fill="both", expand=True)
        
        # History tab
        history_tab = self.analytics_tabs.add("History")
        history_tab.grid_columnconfigure(0, weight=1)
        
        history_frame = ctk.CTkScrollableFrame(history_tab)
        history_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.history_text = ctk.CTkTextbox(history_frame)
        self.history_text.pack(fill="both", expand=True)
        
        # Actions at the bottom
        btn_frame = ctk.CTkFrame(analytics_frame)
        btn_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        export_btn = ctk.CTkButton(
            btn_frame, 
            text="📊 Export Data", 
            command=self._on_export,
            corner_radius=6,
            fg_color="#2b2d42",
            hover_color="#4a4e69"
        )
        export_btn.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        clear_btn = ctk.CTkButton(
            btn_frame, 
            text="🗑️ Clear History", 
            command=self._on_clear,
            corner_radius=6,
            fg_color="#2b2d42",
            hover_color="#4a4e69"
        )
        clear_btn.pack(side="right", padx=5, pady=5, fill="x", expand=True)
    
    def _create_status_bar(self):
        """Create bottom status bar"""
        status_frame = ctk.CTkFrame(self.main_frame, height=40, corner_radius=10)
        status_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        # Status indicators
        self.fps_label = ctk.CTkLabel(
            status_frame, 
            text="FPS: 0",
            font=ctk.CTkFont(size=12),
            padx=10
        )
        self.fps_label.pack(side="left")
        
        self.detection_count_label = ctk.CTkLabel(
            status_frame, 
            text="Detections: 0",
            font=ctk.CTkFont(size=12),
            padx=10
        )
        self.detection_count_label.pack(side="left")
        
        # Session info
        self.session_label = ctk.CTkLabel(
            status_frame, 
            text="Session: --",
            font=ctk.CTkFont(size=12),
            padx=10
        )
        self.session_label.pack(side="left")
        
        self.system_status_label = ctk.CTkLabel(
            status_frame, 
            text="🟢 System Online",
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=10
        )
        self.system_status_label.pack(side="right")
    
    # Event handlers
    def _on_mode_change(self, mode: str):
        """Handle navigation mode change"""
        if self.on_mode_change_callback:
            self.on_mode_change_callback(mode)
    
    def _on_confidence_change(self, value):
        """Handle confidence threshold change"""
        self.confidence_label.configure(text=f"{value:.2f}")
        if self.on_confidence_change_callback:
            self.on_confidence_change_callback(value)
    
    def _on_voice_toggle(self):
        """Handle voice assistant toggle"""
        if self.on_voice_toggle_callback:
            self.on_voice_toggle_callback(self.voice_enabled.get())
    
    def _on_emergency(self):
        """Handle emergency button"""
        if self.on_emergency_callback:
            self.on_emergency_callback()
    
    def _on_export(self):
        """Handle export button"""
        if self.on_export_callback:
            self.on_export_callback()
    
    def _on_clear(self):
        """Handle clear button"""
        if self.on_clear_callback:
            self.on_clear_callback()
    
    def _on_action(self, action: str):
        """Handle quick action buttons"""
        # This can be extended to handle different actions
        logger.info(f"Quick action triggered: {action}")
    
    def _on_closing(self):
        """Handle window closing"""
        if self.on_closing_callback:
            self.on_closing_callback()
        self.root.destroy()
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
    
    # Public methods
    def update_video_frame(self, frame):
        """Update video display with actual camera feed"""
        if self.video_label and frame is not None:
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit the display area
                height, width = frame_rgb.shape[:2]
                display_width = 640
                display_height = 480
                
                # Calculate aspect ratio
                aspect_ratio = width / height
                if aspect_ratio > display_width / display_height:
                    new_width = display_width
                    new_height = int(display_width / aspect_ratio)
                else:
                    new_height = display_height
                    new_width = int(display_height * aspect_ratio)
                
                # Resize frame
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PIL Image
                self.video_image = Image.fromarray(frame_resized)
                
                # Convert to PhotoImage for Tkinter
                self.video_photo = ImageTk.PhotoImage(self.video_image)
                
                # Update the label
                self.video_label.configure(image=self.video_photo, text="")
                
            except Exception as e:
                logger.error(f"Error updating video frame: {e}")
                # Fallback to text display
                self.video_label.configure(image=None, text="Camera Feed Error")
    
    def update_instruction(self, text: str, color: str = "#2ec4b6"):
        """Update instruction text"""
        if self.instruction_label:
            self.instruction_label.configure(text=text, text_color=color)
    
    def update_stats(self, stats_text: str):
        """Update statistics display"""
        if self.stats_text:
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", stats_text)
            self.stats_text.configure(state="disabled")
    
    def update_fps(self, fps: int):
        """Update FPS display"""
        if self.fps_label:
            self.fps_label.configure(text=f"FPS: {fps}")
    
    def update_detection_count(self, count: int):
        """Update detection count"""
        if self.detection_count_label:
            self.detection_count_label.configure(text=f"Detections: {count}")
    
    def update_system_status(self, status: str, color: str = "#2ec4b6"):
        """Update system status"""
        if self.system_status_label:
            self.system_status_label.configure(text=status, text_color=color)
    
    def set_callbacks(self, callbacks: dict):
        """Set callback functions"""
        self.on_mode_change_callback = callbacks.get('mode_change')
        self.on_confidence_change_callback = callbacks.get('confidence_change')
        self.on_voice_toggle_callback = callbacks.get('voice_toggle')
        self.on_emergency_callback = callbacks.get('emergency')
        self.on_export_callback = callbacks.get('export')
        self.on_clear_callback = callbacks.get('clear')
        self.on_closing_callback = callbacks.get('closing')
    
    def run(self):
        """Start the UI main loop"""
        self.root.mainloop()
    
    def destroy(self):
        """Destroy the window"""
        self.root.destroy() 