from PyQt5.QtCore import (QObject, pyqtSignal, QTimer, Qt, pyqtSlot, QThread,
                            QPointF, QRectF, QLineF, QRect)
from pythonosc import udp_client, osc_message_builder
import numpy as np

class OSCclient(QObject):
    """Connects to OSC server to send OSC messages to server
    input: ip of qlab machine, input port number
    default localhost, 53000 (QLab settings)"""
    def __init__(self, ip="127.0.0.1", port=53000):
        QObject.__init__(self)
        self.ip = ip
        self.port = port
        self.client = udp_client.UDPClient(ip, port)
    pyqtSlot(object)
    def emit(self, cuenum):
        msg_raw = f"/cue/{cuenum}/startAndAutoloadNext"
        print(f'{msg_raw} sent')
        msg = osc_message_builder.OscMessageBuilder(msg_raw)
        msg = msg.build()
        self.client.send(msg)
