import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import Image from 'next/image';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Flower Intelligence Chat',
  description: 'Local-first AI chat powered by Flower Intelligence',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-zinc-900`}>
        {children}
        <footer className="w-full text-center text-white bg-transparent">
          <div className="flex items-center justify-center space-x-2 text-lg font-bold">
            <span>
              Powered by{' '}
              <a
                href="https://flower.ai/"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-zinc-200 hover:underline"
              >
                Flower Intelligence
              </a>
            </span>
            <Image src="/fi-icon.png" alt="FlowerLabs Logo" width={24} height={24} />
          </div>
          <div className="mt-2 text-md text-zinc-700">
            <a
              href="https://flower.ai/imprint/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-300 hover:underline mr-4"
            >
              Imprint
            </a>
            <a
              href="https://flower.ai/privacy/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-300 hover:underline"
            >
              Privacy
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
