from xmlrpc.client import ServerProxy
s = ServerProxy("test")
s.Remote.setting()