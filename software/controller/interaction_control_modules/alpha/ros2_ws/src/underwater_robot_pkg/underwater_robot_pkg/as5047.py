import spidev
import time

class AS5X47:
    ANGLE_REG = 0x3FFF
    NOP_REG   = 0x0000

    READ  = 1
    WRITE = 0

    def __init__(self, bus=0, device=1, max_speed=1000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed
        self.spi.mode = 1
        self.spi.bits_per_word = 8

    # ---------- Parity ----------
    @staticmethod
    def _even_parity_15bit(value):
        count = 0
        for i in range(15):
            if value & (1 << i):
                count += 1
        return 1 if (count % 2) != 0 else 0

    # ---------- Frame Builder ----------
    def _build_command(self, address, rw):
        frame = (rw << 14) | (address & 0x3FFF)
        parity = self._even_parity_15bit(frame)
        frame |= (parity << 15)
        return frame

    # ---------- SPI Transfer ----------
    def _transfer16(self, value):
        tx = [(value >> 8) & 0xFF, value & 0xFF]
        rx = self.spi.xfer2(tx)
        return (rx[0] << 8) | rx[1]

    # ---------- Register Read ----------
    def read_register(self, address):
        cmd = self._build_command(address, self.READ)
        nop = self._build_command(self.NOP_REG, self.READ)

        self._transfer16(cmd)         # first transfer (pipeline)
        result = self._transfer16(nop)  # second transfer returns data
        return result

    # ---------- Register Write ----------
    def write_register(self, address, value):
        cmd = self._build_command(address, self.WRITE)

        data_frame = value & 0x3FFF
        parity = self._even_parity_15bit(data_frame)
        data_frame |= (parity << 15)

        self._transfer16(cmd)
        self._transfer16(data_frame)

    # ---------- Angle Read ----------
    def read_angle(self):
        raw = self.read_register(self.ANGLE_REG)
        # print(f"RAW FRAME: 0x{raw:04X}")
        angle_raw = raw & 0x3FFF
        return angle_raw * 360.0 / 16384.0
    
    # def read_angle_raw_test(self):
    #     response = self.spi.xfer2([0xFF, 0xFF])
    #     print(response)

    def close(self):
        self.spi.close()



if __name__ == "__main__":
    encoder = AS5X47(bus=0, device=1)  # match your /dev/spidev0.1
    # encoder = AS5X47(bus=10, device=0)  # match your /dev/spidev0.1

    try:
        while True:
            angle = encoder.read_angle()
            print(f"Angle: {angle:.3f} deg")
            time.sleep(0.1)

    except KeyboardInterrupt:
        encoder.close()