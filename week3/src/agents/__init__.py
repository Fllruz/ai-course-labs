# -*- coding: utf-8 -*-
"""Агенты многоагентной системы."""

from agents.base_agent import BaseAgent, AgentConfig, AgentState
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.writer_agent import WriterAgent
from agents.smart_camera_agent import SmartCameraAgent

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'AgentState',
    'ResearcherAgent',
    'AnalystAgent',
    'WriterAgent',
    'SmartCameraAgent',
]