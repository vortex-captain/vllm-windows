// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io;
use std::mem::MaybeUninit;

use windows_sys::Win32::Networking::WinSock::{WSACleanup, WSADATA, WSAStartup};

pub struct Winsock;

impl Winsock {
    pub fn initialize() -> io::Result<Self> {
        const VERSION: u16 = 0x0202;
        let mut data = MaybeUninit::<WSADATA>::uninit();
        let result = unsafe { WSAStartup(VERSION, data.as_mut_ptr()) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
        Ok(Self)
    }
}

impl Drop for Winsock {
    fn drop(&mut self) {
        unsafe { WSACleanup() };
    }
}
