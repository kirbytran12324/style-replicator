'use client';

import Link from 'next/link';
import { Home, Settings, BrainCircuit, Images, Plus, Sparkles } from 'lucide-react';
import { FaXTwitter, FaDiscord, FaYoutube } from 'react-icons/fa6';

const Sidebar = () => {
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Train', href: '/train', icon: Plus }, // Renamed & Repointed
    { name: 'Generate', href: '/generate', icon: Sparkles }, // New!
    { name: 'Job History', href: '/jobs', icon: BrainCircuit },
    { name: 'Datasets', href: '/datasets', icon: Images },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const socialsBoxClass =
    'flex flex-col items-center justify-center p-1 hover:bg-gray-800 rounded-lg transition-colors';
  const socialIconClass = 'w-5 h-5 text-gray-400 hover:text-white';

  return (
    <div className="flex flex-col w-59 bg-gray-900 text-gray-100 h-full border-r border-gray-800">
      <div className="px-4 py-4">
        <Link href="/dashboard" className="text-l flex items-center">
          {/* Ensure this image exists in /public or remove it */}
          <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-7 mr-3 inline" />
          <span className="font-bold uppercase tracking-wider">AI</span>
          <span className="ml-2 uppercase text-gray-400 text-xs mt-1">Toolkit</span>
        </Link>
      </div>
      
      <nav className="flex-1 px-2 py-4">
        <ul className="space-y-1">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2.5 text-sm font-medium text-gray-300 hover:bg-gray-800 hover:text-white rounded-lg transition-all group"
              >
                <item.icon className="w-5 h-5 mr-3 text-gray-400 group-hover:text-blue-400 transition-colors" />
                {item.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer / Socials */}
      {/*<div className="p-4 border-t border-gray-800">*/}
      {/*  <a*/}
      {/*    href="https://ostris.com/support"*/}
      {/*    target="_blank"*/}
      {/*    rel="noreferrer"*/}
      {/*    className="flex items-center space-x-2 text-xs text-gray-500 hover:text-gray-300 mb-4 transition-colors"*/}
      {/*  >*/}
      {/*    <div className="w-5 h-5 text-red-500">*/}
      {/*       <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>*/}
      {/*    </div>*/}
      {/*    <span>Support Development</span>*/}
      {/*  </a>*/}

      {/*  <div className="grid grid-cols-3 gap-2">*/}
      {/*    <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialsBoxClass}>*/}
      {/*      <FaDiscord className={socialIconClass} />*/}
      {/*    </a>*/}
      {/*    <a href="https://www.youtube.com/@ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>*/}
      {/*      <FaYoutube className={socialIconClass} />*/}
      {/*    </a>*/}
      {/*    <a href="https://x.com/ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>*/}
      {/*      <FaXTwitter className={socialIconClass} />*/}
      {/*    </a>*/}
      {/*  </div>*/}
      {/*</div>*/}
    </div>
  );
};

export default Sidebar;