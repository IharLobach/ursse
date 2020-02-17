# Read_PTU.py    Read PicoQuant Unified Histogram Files
# This is demo code. Use at your own risk. No warranties.
# Keno Goertz, PicoQUant GmbH, February 2018

# Note that marker events have a lower time resolution and may therefore appear
# in the file slightly out of order with respect to regular (photon) event records.
# This is by design. Markers are designed only for relatively coarse
# synchronization requirements such as image scanning.

# T Mode data are written to an output file [filename]
# We do not keep it in memory because of the huge amout of memory
# this would take in case of large files. Of course you can change this,
# e.g. if your files are not too big.
# Otherwise it is best process the data on the fly and keep only the results.

import time
import sys
import struct
import io
import numpy as np
import pandas as pd

# Tag Types
tyEmpty8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
tyBool8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
tyInt8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
tyBitSet64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
tyColor8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
tyFloat8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

# Record types
rtPicoHarpT3     = struct.unpack(">i", bytes.fromhex('00010303'))[0]
rtPicoHarpT2     = struct.unpack(">i", bytes.fromhex('00010203'))[0]
rtHydraHarpT3    = struct.unpack(">i", bytes.fromhex('00010304'))[0]
rtHydraHarpT2    = struct.unpack(">i", bytes.fromhex('00010204'))[0]
rtHydraHarp2T3   = struct.unpack(">i", bytes.fromhex('01010304'))[0]
rtHydraHarp2T2   = struct.unpack(">i", bytes.fromhex('01010204'))[0]
rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
rtMultiHarpNT3   = struct.unpack(">i", bytes.fromhex('00010307'))[0]
rtMultiHarpNT2   = struct.unpack(">i", bytes.fromhex('00010207'))[0]



class HydraHarpFile:
    def __init__(self,inputfile,safemode=False):
        self.inputfile = open(inputfile, "rb")
        self.outputfile = ""
        self.oflcorrection = 0
        self.dlen = 0
        self.TimeTags=None
        self.__read(safemode)

    def __read(self,safemode):
        # Check if self.inputfile is a valid PTU file
        # Python strings don't have terminating NULL characters, so they're stripped
        magic = self.inputfile.read(8).decode("utf-8").strip('\0')
        if magic != "PQTTTR":
            self.inputfile.close()
            raise Exception("ERROR: Magic invalid, this is not a PTU file.")

        version = self.inputfile.read(8).decode("utf-8").strip('\0')
        self.outputfile+=("Tag version: %s\n" % version)
        self.__read_header()  #reading tag values

        # get important variables from headers
        self.numRecords = self.tagValues[self.tagNames.index("TTResult_NumberOfRecords")]
        globRes = self.tagValues[self.tagNames.index("MeasDesc_GlobalResolution")]
        #print("Writing %d records, this may take a while..." % self.numRecords)

        self.outputfile+=("\n-----------------------\n")
        self.recordType = self.tagValues[self.tagNames.index("TTResultFormat_TTTRRecType")]

        if safemode:
            self.__read_time_tags_hydraharp_demo()
        else:
            self.__my_read_time_tags()


        self.TimeTags  = pd.DataFrame({"Channel":self.ChannelArr,"TimeTag":self.TimeTagArr})

        self.inputfile.close()

    def __read_header(self):
        # Write the header data to self.outputfile and also save it in memory.
        # There's no do ... while in Python, so an if statement inside the while loop
        # breaks out of it
        tagDataList = []    # Contains tuples of (tagName, tagValue)
        while True:
            tagIdent = self.inputfile.read(32).decode("utf-8").strip('\0')
            tagIdx = struct.unpack("<i", self.inputfile.read(4))[0]
            tagTyp = struct.unpack("<i", self.inputfile.read(4))[0]
            if tagIdx > -1:
                evalName = tagIdent + '(' + str(tagIdx) + ')'
            else:
                evalName = tagIdent
            self.outputfile+=("\n%-40s" % evalName)
            if tagTyp == tyEmpty8:
                self.inputfile.read(8)
                self.outputfile+=("<empty Tag>")
                tagDataList.append((evalName, "<empty Tag>"))
            elif tagTyp == tyBool8:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                if tagInt == 0:
                    self.outputfile+=("False")
                    tagDataList.append((evalName, "False"))
                else:
                    self.outputfile+=("True")
                    tagDataList.append((evalName, "True"))
            elif tagTyp == tyInt8:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                self.outputfile+=("%d" % tagInt)
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyBitSet64:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                self.outputfile+=("{0:#0{1}x}".format(tagInt,18))
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyColor8:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                self.outputfile+=("{0:#0{1}x}".format(tagInt,18))
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyFloat8:
                tagFloat = struct.unpack("<d", self.inputfile.read(8))[0]
                self.outputfile+=("%-3E" % tagFloat)
                tagDataList.append((evalName, tagFloat))
            elif tagTyp == tyFloat8Array:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                self.outputfile+=("<Float array with %d entries>" % tagInt/8)
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyTDateTime:
                tagFloat = struct.unpack("<d", self.inputfile.read(8))[0]
                tagTime = int((tagFloat - 25569) * 86400)
                tagTime = time.gmtime(tagTime)
                self.outputfile+=(time.strftime("%a %b %d %H:%M:%S %Y", tagTime))
                tagDataList.append((evalName, tagTime))
            elif tagTyp == tyAnsiString:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                tagString = self.inputfile.read(tagInt).decode("utf-8").strip("\0")
                self.outputfile+=("%s" % tagString)
                tagDataList.append((evalName, tagString))
            elif tagTyp == tyWideString:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                tagString = self.inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
                self.outputfile+=(tagString)
                tagDataList.append((evalName, tagString))
            elif tagTyp == tyBinaryBlob:
                tagInt = struct.unpack("<q", self.inputfile.read(8))[0]
                self.outputfile+=("<Binary blob with %d bytes>" % tagInt)
                tagDataList.append((evalName, tagInt))
            else:
                print("ERROR: Unknown tag type")
                exit(0)
            if tagIdent == "Header_End":
                break

        # Reformat the saved data for easier access
        self.tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
        self.tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]

    def __read_time_tags_hydraharp_demo(self):
        self.ChannelArr = np.zeros(self.numRecords,np.uint8)
        self.TimeTagArr = np.zeros(self.numRecords,np.uint64)
        if self.recordType == rtPicoHarpT2:
            self.isT2 = True
            print("PicoHarp T2 data")
            self.outputfile+=("PicoHarp T2 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ps\n")
            self.__readPT2()
        elif self.recordType == rtPicoHarpT3:
            self.isT2 = False
            print("PicoHarp T3 data")
            self.outputfile+=("PicoHarp T3 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ns dtime\n")
            self.__readPT3()
        elif self.recordType == rtHydraHarpT2:
            self.isT2 = True
            print("HydraHarp V1 T2 data")
            self.outputfile+=("HydraHarp V1 T2 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ps\n")
            self.__readHT2(1)
        elif self.recordType == rtHydraHarpT3:
            self.isT2 = False
            print("HydraHarp V1 T3 data")
            self.outputfile+=("HydraHarp V1 T3 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ns dtime\n")
            self.__readHT3(1)
        elif self.recordType == rtHydraHarp2T2:
            self.isT2 = True
            print("HydraHarp V2 T2 data")
            self.outputfile+=("HydraHarp V2 T2 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ps\n")
            self.__readHT2(2)
        elif self.recordType == rtHydraHarp2T3:
            self.isT2 = False
            print("HydraHarp V2 T3 data")
            self.outputfile+=("HydraHarp V2 T3 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ns dtime\n")
            self.__readHT3(2)
        elif self.recordType == rtTimeHarp260NT3:
            self.isT2 = False
            print("TimeHarp260N T3 data")
            self.outputfile+=("TimeHarp260N T3 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ns dtime\n")
            self.__readHT3(2)
        elif self.recordType == rtTimeHarp260NT2:
            self.isT2 = True
            print("TimeHarp260N T2 data")
            self.outputfile+=("TimeHarp260N T2 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ps\n")
            self.__readHT2(2)
        elif self.recordType == rtTimeHarp260PT3:
            self.isT2 = False
            print("TimeHarp260P T3 data")
            self.outputfile+=("TimeHarp260P T3 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ns dtime\n")
            self.__readHT3(2)
        elif self.recordType == rtTimeHarp260PT2:
            self.isT2 = True
            print("TimeHarp260P T2 data")
            self.outputfile+=("TimeHarp260P T2 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ps\n")
            self.__readHT2(2)
        elif self.recordType == rtMultiHarpNT3:
            self.isT2 = False
            print("MultiHarp150N T3 data")
            self.outputfile+=("MultiHarp150N T3 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ns dtime\n")
            self.__readHT3(2)
        elif self.recordType == rtMultiHarpNT2:
            self.isT2 = True
            print("MultiHarp150N T2 data")
            self.outputfile+=("MultiHarp150N T2 data\n")
            self.outputfile+=("\nrecord# chan   nsync truetime/ps\n")
            self.__readHT2(2)
        else:
            raise Exception("ERROR: Unknown record type")
        self.ChannelArr = self.ChannelArr[self.TimeTagArr!=0]
        self.TimeTagArr = self.TimeTagArr[self.TimeTagArr!=0]

    def __my_read_time_tags(self):
        records_4uint8 = np.flip(np.fromfile(self.inputfile, np.uint8).reshape((-1, 4)), 1)
        records_32bits = np.unpackbits(records_4uint8).reshape((-1, 32))
        if records_32bits.shape[0] != self.numRecords:
            raise Exception("Number of unpacked records doesn't match with that specified in the header!")
        sT, channel_bits, timetag_bits = np.hsplit(records_32bits, np.array([1, 7]))
        special = sT.T[0]
        channel = np.right_shift(np.packbits(channel_bits, axis=-1), 2).T[0]
        timetag_bits = np.concatenate((np.zeros((self.numRecords, 7), np.uint8), timetag_bits), axis=1)
        timetag = np.packbits(timetag_bits.reshape(-1, 4, 8)[:, ::-1]).view(np.uint32)
        T2WRAPAROUND_V2 = 33554432
        ofl = np.where(channel == 0x3F,1,0)
        delta_oflcorrection = ofl * timetag * T2WRAPAROUND_V2
        oflcorrection = np.cumsum(delta_oflcorrection)
        self.TimeTagArr = (oflcorrection + timetag)[channel != 0x3F]
        self.ChannelArr = (channel + 1 - special)[channel != 0x3F]

    def __gotOverflow(self, count, recNum):
        raise NotImplementedError("Got Overflow record! Did not expect it.")
        # self.outputfile+=("%u OFL * %2x\n" % (recNum, count))

    def __gotMarker(self, timeTag, markers, recNum):
        raise NotImplementedError("Got Marker record! Did not expect it.")
        # self.outputfile+=("%u MAR %2x %u\n" % (recNum, markers, timeTag))

    def __gotPhoton(self, timeTag, channel, dtime, recNum):
        if self.isT2:
            self.ChannelArr[recNum] = channel
            self.TimeTagArr[recNum] = timeTag
            # self.outputfile+=("%u CHN %1x %u %8.0lf\n" % (recNum, channel, timeTag,\
            #                  (timeTag * globRes * 1e12)))
        else:
            raise NotImplementedError("Expected isT2==True, got isT2==False.")
            # self.outputfile+=("%u CHN %1x %u %8.0lf %10u\n" % (recNum, channel,\
            #                  timeTag, (timeTag * globRes * 1e9), dtime))

    def __readPT3(self):
        T3WRAPAROUND = 65536
        for recNum in range(0, self.numRecords):
            # The data is stored in 32 bits that need to be divided into smaller
            # groups of bits, with each group of bits representing a different
            # variable. In this case, channel, dtime and nsync. This can easily be
            # achieved by converting the 32 bits to a string, dividing the groups
            # with simple array slicing, and then converting back into the integers.
            try:
                recordData = "{0:0{1}b}".format(struct.unpack("<I", self.inputfile.read(4))[0], 32)
            except Exception as e:
                raise e("The file ended earlier than expected, at record %d/%d." \
                      % (recNum, self.numRecords))

            channel = int(recordData[0:4], base=2)
            dtime = int(recordData[4:16], base=2)
            nsync = int(recordData[16:32], base=2)
            if channel == 0xF:  # Special record
                if dtime == 0:  # Not a marker, so overflow
                    self.__gotOverflow(1, recNum)
                    self.oflcorrection += T3WRAPAROUND
                else:
                    truensync = self.oflcorrection + nsync
                    self.__gotMarker(truensync, dtime, recNum)
            else:
                if channel == 0 or channel > 4:  # Should not occur
                    print("Illegal Channel: #%1d %1u" % (self.dlen, channel))
                    self.outputfile+=("\nIllegal channel ")
                truensync = self.oflcorrection + nsync
                self.__gotPhoton(truensync, channel, dtime, recNum)
                self.dlen += 1
            if recNum % 100000 == 0:
                sys.stdout.write("\rProgress: %.1f%%" % (float(recNum) * 100 / float(self.numRecords)))
                sys.stdout.flush()

    def __readPT2(self):
        T2WRAPAROUND = 210698240
        for recNum in range(0, self.numRecords):
            try:
                recordData = "{0:0{1}b}".format(struct.unpack("<I", self.inputfile.read(4))[0], 32)
            except Exception as e:
                raise e("The file ended earlier than expected, at record %d/%d." \
                      % (recNum, self.numRecords))

            channel = int(recordData[0:4], base=2)
            time = int(recordData[4:32], base=2)
            if channel == 0xF:  # Special record
                # lower 4 bits of time are marker bits
                markers = int(recordData[28:32], base=2)
                if markers == 0:  # Not a marker, so overflow
                    self.__gotOverflow(1, recNum)
                    self.oflcorrection += T2WRAPAROUND
                else:
                    # Actually, the lower 4 bits for the time aren't valid because
                    # they belong to the marker. But the error caused by them is
                    # so small that we can just ignore it.
                    truetime = self.oflcorrection + time
                    self.__gotMarker(truetime, markers, recNum)
            else:
                if channel > 4:  # Should not occur
                    print("Illegal Channel: #%1d %1u" % (recNum, channel))
                    self.outputfile+=("\nIllegal channel ")
                truetime = self.oflcorrection + time
                self.__gotPhoton(truetime, channel, time, recNum)
            if recNum % 100000 == 0:
                sys.stdout.write("\rProgress: %.1f%%" % (float(recNum) * 100 / float(self.numRecords)))
                sys.stdout.flush()

    def __readHT3(self, version):
        T3WRAPAROUND = 1024
        for recNum in range(0, self.numRecords):
            try:
                recordData = "{0:0{1}b}".format(struct.unpack("<I", self.inputfile.read(4))[0], 32)
            except Exception as e:
                raise e("The file ended earlier than expected, at record %d/%d." \
                      % (recNum, self.numRecords))

            special = int(recordData[0:1], base=2)
            channel = int(recordData[1:7], base=2)
            dtime = int(recordData[7:22], base=2)
            nsync = int(recordData[22:32], base=2)
            if special == 1:
                if channel == 0x3F:  # Overflow
                    # Number of overflows in nsync. If 0 or old version, it's an
                    # old style single overflow
                    if nsync == 0 or version == 1:
                        self.oflcorrection += T3WRAPAROUND
                        self.__gotOverflow(1, recNum)
                    else:
                        self.oflcorrection += T3WRAPAROUND * nsync
                        self.__gotOverflow(nsync, recNum)
                if channel >= 1 and channel <= 15:  # markers
                    truensync = self.oflcorrection + nsync
                    self.__gotMarker(truensync, channel, recNum)
            else:  # regular input channel
                truensync = self.oflcorrection + nsync
                self.__gotPhoton(truensync, channel, dtime, recNum)
            if recNum % 100000 == 0:
                sys.stdout.write("\rProgress: %.1f%%" % (float(recNum) * 100 / float(self.numRecords)))
                sys.stdout.flush()

    def __readHT2(self, version):
        T2WRAPAROUND_V1 = 33552000
        T2WRAPAROUND_V2 = 33554432
        for recNum in range(0, self.numRecords):
            try:
                recordData = "{0:0{1}b}".format(struct.unpack("<I", self.inputfile.read(4))[0], 32)
            except Exception as e:
                raise e("The file ended earlier than expected, at record %d/%d." \
                      % (recNum, self.numRecords))

            special = int(recordData[0:1], base=2)
            channel = int(recordData[1:7], base=2)
            timetag = int(recordData[7:32], base=2)
            if special == 1:
                if channel == 0x3F:  # Overflow
                    # Number of overflows in nsync. If old version, it's an
                    # old style single overflow
                    if version == 1:
                        self.oflcorrection += T2WRAPAROUND_V1
                        self.__gotOverflow(1, recNum)
                    else:
                        if timetag == 0:  # old style overflow, shouldn't happen
                            self.oflcorrection += T2WRAPAROUND_V2
                            self.__gotOverflow(1, recNum)
                        else:
                            self.oflcorrection += T2WRAPAROUND_V2 * timetag
                if channel >= 1 and channel <= 15:  # markers
                    truetime = self.oflcorrection + timetag
                    self.__gotMarker(truetime, channel, recNum)
                if channel == 0:  # sync
                    truetime = self.oflcorrection + timetag
                    self.__gotPhoton(truetime, 0, 0, recNum)
            else:  # regular input channel
                truetime = self.oflcorrection + timetag
                self.__gotPhoton(truetime, channel + 1, 0, recNum)
            if recNum % 100000 == 0:
                sys.stdout.write("\rProgress: %.1f%%" % (float(recNum) * 100 / float(self.numRecords)))
                sys.stdout.flush()



if __name__=="__main__":
    f = HydraHarpFile("S_7p5MHz_Ch4_3MHz_000.ptu",safemode=False)
    True

